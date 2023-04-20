import argparse
import pickle
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
#from torchsummary import summary
from config import config_training_setup
from src.metrics.stream_metrics import StreamSegMetrics, RMSE, AEPE
from src.losses import cross_entropy, InvHuberLoss, encode_target
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.cs_laf_model_utils import load_network
from src.early_stopping import EarlyStopping
from torch.utils.data import DataLoader,random_split 
from torch.utils.tensorboard import SummaryWriter
import os


def training_routine(config,early_stopping=True):
    """Start OoD Training"""
    print("START OOD TRAINING")
    params = config.params
    alpha=params.pareto_alpha
    roots = config.roots
    dataset = config.dataset()
    architecture = config.architecture
    result_dir=os.path.join(roots.io_root, 'results')
    save_dir_data = os.path.join(result_dir,'entropy_counts_per_pixel')
    save_dir_metrics=os.path.join(result_dir,'metrics')
    start_epoch = params.training_starting_epoch
    epochs = params.num_training_epochs
    start = time.time()

    """Initialize model"""
    if start_epoch == 0:
        network, checkpoint = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt,architecture = architecture)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network, checkpoint = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename),architecture = architecture)
    network = network.cuda(0)
    optimizer = optim.Adam(network.parameters(), lr=params.learning_rate)
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.9,patience=5,threshold=1,threshold_mode='abs',verbose=True)#3,1##0.95,10,0.5
    if start_epoch != 0:
        start_epoch=checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['schedular_state_dict'])
        print('\n current lr is', optimizer.param_groups[0]['lr'])

    transform = Compose([RandomHorizontalFlip(), RandomCrop(params.crop_size), ToTensor(),
                         Normalize(dataset.mean, dataset.std)])

    trainloader = config.dataset('train', transform,roots.cs_root, roots.laf_root,shuffle=True)#, params.ood_subsampling_factor)
    train_loader,val_loader= random_split(trainloader,lengths=[int(0.9*len(trainloader)),(len(trainloader)-int(0.9*len(trainloader)))],generator=torch.Generator().manual_seed(42))
    train_data=DataLoader(train_loader,batch_size=params.batch_size,shuffle=True,drop_last=True,num_workers=8)
    val_data=DataLoader(val_loader,batch_size=params.batch_size,shuffle=True,drop_last=True,num_workers=8)

    save_summary=os.path.join(roots.io_root,'summary')
    writer=SummaryWriter(save_summary,flush_secs=30)
    if not os.path.exists(save_summary):
        os.makedirs(save_summary)
        print('\n Summary path directory is created at',str(save_summary))

    print('No. of train batches is {} and No. of Validation Batches is {}'.format(len(train_data),len(val_data)))
    #Task-balancing method
    if config.WEIGHTING_METHOD == 'dwa':
        train_task = 2
        # Dynamic weighting Average
        T =2.0 #temperature used in dwa
        lambda_weight=np.ones([epochs+start_epoch,2])
        save_losses = os.path.join(roots.io_root, 'dwa_losses.csv')
        save_weights = os.path.join(roots.io_root ,'dwa_tasks_weights.csv')

    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""
        
        print('\nEpoch {}/{}'.format(epoch + 1, start_epoch + epochs))
        pattern = "epoch_" + str(epoch+1) + "_alpha_" + str(alpha)
        save_path_data = os.path.join(save_dir_data, pattern + ".p")
        save_metrics_data=os.path.join(save_dir_metrics, pattern + ".p")
        if not os.path.exists(save_path_data) :
            save_dir = os.path.dirname(save_path_data)
            if not os.path.exists(save_dir):
                print("\n Create directory", save_dir)
                os.makedirs(save_dir)
        if not os.path.exists(save_metrics_data) :
            save_metrics_dir=os.path.dirname(save_metrics_data)
            if not os.path.exists(save_metrics_dir):
                print("\n Create directory for metrics",save_metrics_dir)
                os.makedirs(save_metrics_dir)

        ##dynamic Weighting average
        if config.WEIGHTING_METHOD == 'dwa' :
            if epoch == 0 or epoch == 1:
                lambda_weight[epoch,0] = 0.98
                lambda_weight[epoch,1] = 0.02
            else:
                df_saved_losses = pd.read_csv(save_losses)
                saved_train_loss = np.array(df_saved_losses.iloc[:,[1,2]])
                w = []
                for i in range(train_task): #train_seg_loss and train_disp_loss
                    w += [saved_train_loss[epoch - 1][i] / saved_train_loss[epoch - 2][i]]
                w = torch.softmax(torch.tensor(w) / T, dim=0)
                lambda_weight[epoch] =  2 * w.numpy()
                print('W_1 is  {:3f} and W_2 is {:3f}'.format(lambda_weight[epoch][0],lambda_weight[epoch][1]))
            if not os.path.exists (save_weights):
                with open (save_weights, 'w', newline='') as wfile:
                    pd.DataFrame([(epoch+1,lambda_weight[epoch][0],lambda_weight[epoch][1])],columns=['epoch','w1','w2']).to_csv(wfile,index= False)
            else:
                pd.DataFrame([(epoch+1,lambda_weight[epoch][0],lambda_weight[epoch][1])]).to_csv(save_weights, mode='a', index=False,header= not os.path.exists(save_weights))
        
        ## Uncertainty weighting
        elif config.WEIGHTING_METHOD in ['uw','adamtnet']:
            seg_weight_train_epoch = 0.0
            disp_weight_train_epoch =0.0
            seg_uncer_train_epoch = 0.0
            disp_uncer_train_epoch = 0.0

        train_loss = 0.0
        train_seg_loss = 0.0
        train_disp_loss = 0.0
        train_cross_ent = 0.0
      
        train_data = tqdm(train_data,position=0,leave=True)
        network.train()
        train_metrics = StreamSegMetrics(n_classes=dataset.num_classes)
        train_rmse = RMSE(ignore_val=0)
        train_aepe = AEPE(ignore_val=0)
        train_metrics.reset()
        train_rmse.reset()
        train_aepe.reset()
        weights = [1,1]
        for i,(x, target,dmaps,ids) in enumerate(train_data):
            optimizer.zero_grad() 
            target = target.cuda(0)
            logits,disp = network(x.cuda(0))
            y = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
                              ignore_train_ind=dataset.void_ind, ood_ind=dataset.ood_ind).cuda(0)
            cs_img_batch = 0
            seg_loss_batch = 0.0                 
            for j,id in enumerate(ids):
                if id == 0:
                    train_metrics.update(label_trues=target[j],label_preds=logits[j],train=True)
                    seg_loss_batch += cross_entropy (logits = logits[j],targets=y[j])
                    cs_img_batch += 1
            train_rmse.update(pred = disp,gt=dmaps,train=True)
            train_aepe.update(pred = disp,gt= dmaps, train= True)
            seg_loss = seg_loss_batch / cs_img_batch
            train_cross_ent += seg_loss.item()

            dmaps = dmaps.cuda(0)
            disp_loss=InvHuberLoss(disp,target=dmaps,ignore_index=0)

            if config.WEIGHTING_METHOD == 'dwa':
                loss =(lambda_weight[epoch][0] * seg_loss) + (lambda_weight[epoch][1] * disp_loss)
                loss.backward()

            elif config.WEIGHTING_METHOD == 'uw':
                get_loss_params= network.module.get_loss_params
                seg_log_var,disp_log_var = get_loss_params()[0], get_loss_params()[1] 
                seg_uncer_train_epoch += seg_log_var
                disp_uncer_train_epoch += disp_log_var
                seg_weight = torch.exp(- seg_log_var)
                disp_weight = 0.5 * torch.exp(- disp_log_var)
                seg_weight_train_epoch += seg_weight
                disp_weight_train_epoch += disp_weight
                seg_loss = ((1 / torch.exp(seg_log_var)) * seg_loss) + (seg_log_var/2)
                disp_loss = (1 /(2 * torch.exp(disp_log_var)) * disp_loss) + (disp_log_var/2)
                loss = seg_loss+ disp_loss
                loss.backward()
                writer.add_scalars('Train/Uncertainty_iter',{'segmentation':seg_log_var,'disparity': disp_log_var},epoch*len(train_data)+i)

            elif config.WEIGHTING_METHOD == 'adamtnet':
                loss = weights[0] * seg_loss + weights[1] * disp_loss
                loss.backward()
                grad_seg = network.module.final[3].weight.grad # (256,256,3,3)
                grad_disp = network.module.final_depth[3].weight.grad  # (256,256,3,3)
                norm_grad_seg = torch.mean(F.normalize(grad_seg))
                norm_grad_disp = torch.mean(F.normalize(grad_disp))
                seg_weight = norm_grad_seg / (norm_grad_seg + norm_grad_disp)
                disp_weight = norm_grad_disp / (norm_grad_seg + norm_grad_disp)
                seg_weight = seg_weight.reshape((1,))
                disp_weight = disp_weight.reshape((1,))
                w = torch.cat([seg_weight,disp_weight])
                w = torch.softmax(w, dim=0)
                weights[0],weights[1] = w[0], w[1]
                seg_weight_train_epoch += w[0]
                disp_weight_train_epoch += w[1]
                writer.add_scalars('Train/weights_iter',{'segmentation':w[0],'disparity': w[1]},epoch*len(train_data)+i)
              
            elif config.WEIGHTING_METHOD == 'Fix':
                loss= 0.98 * seg_loss + 0.02 * disp_loss
                loss.backward()
            else:
                print('please specify correct weighting method')

            optimizer.step()
            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_disp_loss += disp_loss.item()
            train_data.set_description('Train : Epoch %i Loss : %.8f'%(epoch+1,loss))
            train_data.refresh()
            
        score=train_metrics.get_results()
        #print('class_iou :', score['Class IoU']) 
        rmse_train=train_rmse.val()
        aepe_train = train_aepe.val()
        miou=score["Mean IoU"] 
        scheduler.step(miou)
        lr=np.array([x['lr'] for x in optimizer.param_groups ])
        print('\n New lr is {}'.format(lr))    
        print(' *'*10,' Training Results' , '*'*10) 
        print('\n mIoU_train is {:5f} and RMSE_train s {:5f}  and AEPE_Train is {:5f} '.format(miou,rmse_train,aepe_train))
        
        avg_train_loss=train_loss / len(train_data)
        avg_train_seg_loss=train_seg_loss / len(train_data)
        avg_train_disp_loss=train_disp_loss / len(train_data)
        
        if config.WEIGHTING_METHOD == 'uw':
            ave_train_seg_weight = seg_weight_train_epoch / len(train_data)
            ave_train_disp_weight = disp_weight_train_epoch / len(train_data)
            ave_train_seg_uncer = seg_uncer_train_epoch / len(train_data)
            ave_train_disp_uncer = disp_uncer_train_epoch / len (train_data)
            writer.add_scalars('Train/weights',{'segmentation':ave_train_seg_weight,'disparity': ave_train_disp_weight},epoch+1)
            writer.add_scalars('Train/uncertainty',{'segmentation':ave_train_seg_uncer,'disparity': ave_train_disp_uncer},epoch+1)
            #print('Train : Segmentation Weight is {:2f} and Disparity weight is {:2f}'.format(ave_train_seg_weight,ave_train_disp_weight))
        elif config.WEIGHTING_METHOD == 'adamtnet':
            ave_train_seg_weight = seg_weight_train_epoch / len(train_data)
            ave_train_disp_weight = disp_weight_train_epoch / len(train_data)          
            writer.add_scalars('Train/weights',{'segmentation':ave_train_seg_weight,'disparity': ave_train_disp_weight},epoch+1)
        elif config.WEIGHTING_METHOD == 'dwa':
                      
            writer.add_scalars('Train/weights',{'segmentation':lambda_weight[epoch][0],'disparity': lambda_weight[epoch][1]},epoch+1)

        writer.add_scalar('Train/seg_loss',avg_train_seg_loss,epoch+1)
        writer.add_scalar('Train/disparity_loss',avg_train_disp_loss,epoch+1)
        writer.add_scalar('Train/total_loss',avg_train_loss,epoch+1)
        writer.add_scalar('Train/mIoU',miou,epoch+1)
        writer.add_scalar('Train/RMSE',rmse_train,epoch+1)
        writer.add_scalar('Train/AEPE',aepe_train,epoch+1)
        writer.add_scalar('Train / learning_rate',lr,epoch+1)
        
        #validation 
        network.eval()
        val_loss=0.0
        val_seg_loss=0.0
        val_disp_loss=0.0
        val_cross_ent=0.0
        val_data=tqdm(val_data,position=0,leave=True)

        with torch.no_grad():           
            val_metrics = StreamSegMetrics(n_classes=dataset.num_classes)
            val_metrics.reset()
            val_rmse = RMSE(ignore_val=0)
            val_aepe = AEPE(ignore_val = 0)
            val_rmse.reset()
            val_aepe.reset()
            for i,(x,target,dmaps,ids) in enumerate(val_data):
                logits,disp = network(x.cuda(0))
                y = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
                                ignore_train_ind=dataset.void_ind, ood_ind=dataset.ood_ind).cuda(0) 
                cs_img_batch = 0
                seg_loss_batch = 0.0 
                for i,id in enumerate(ids):
                    if id == 0:
                        val_metrics.update(label_trues=target[i],label_preds=logits[i],train=True)
                        seg_loss_batch += cross_entropy (logits = logits[i],targets=y[i])
                        cs_img_batch+= 1
                val_rmse.update(pred=disp,gt=dmaps)
                val_aepe.update (pred=disp,gt=dmaps)
                seg_loss = seg_loss_batch / cs_img_batch
                val_cross_ent+=seg_loss.item()
                dmaps=dmaps.cuda(0)
                disp_loss=InvHuberLoss(disp,target=dmaps,ignore_index=0)

                if config.WEIGHTING_METHOD == 'dwa':
                    loss=(lambda_weight[epoch][0] * seg_loss) + (lambda_weight[epoch][1] * disp_loss)
                
                elif config.WEIGHTING_METHOD == 'uw':
                    seg_loss = ((1 / torch.exp(ave_train_seg_uncer)) * seg_loss + ave_train_seg_uncer/2)
                    disp_loss = 1 /(2 * torch.exp(ave_train_disp_uncer)) * disp_loss + ave_train_disp_uncer/2
                    loss = seg_loss+ disp_loss

                elif config.WEIGHTING_METHOD == 'adamtnet':
                    loss = ave_train_seg_weight * seg_loss + ave_train_disp_weight * disp_loss

                elif config.WEIGHTING_METHOD == 'Fix':
                    loss= 0.98 * seg_loss + 0.02 * disp_loss
                else:
                    print('please specify correct weighting method')
                
                val_data.set_description('Validation : Epoch %i Loss : %.8f'%(epoch+1,loss))
                val_data.refresh()
                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_disp_loss += disp_loss.item()
                
            score=val_metrics.get_results()
            val_miou=score["Mean IoU"]   
            print('*'*10,' Validation Results' , '*'*10) 
            rmse_val = val_rmse.val()
            aepe_val = val_aepe.val ()
            print('\n mIoU_Val is {:5f} and RMSE_Val is {:5f} and AEPE_Val is {:5f}'.format(val_miou,rmse_val,aepe_val))
          
        performance_metrics={'mIoU_train' : miou,'RMSE_Train':rmse_train,'AEPE_Train':aepe_train,
                                'mIoU_val':val_miou,'RMSE_Val':rmse_val,'AEPE_val':val_aepe}
        pickle.dump(performance_metrics,open(save_metrics_data, "wb"))

        avg_val_loss = val_loss / len(val_data)
        avg_val_seg_loss = val_seg_loss / len(val_data)
        avg_val_disp_loss = val_disp_loss / len(val_data)

        if config.WEIGHTING_METHOD == 'dwa':
            #save loss of every epoch in to csv
            data_losses = [( epoch + 1, avg_train_seg_loss, avg_train_disp_loss, avg_train_loss, avg_val_seg_loss, avg_val_disp_loss, avg_val_loss )]
            df_losses = pd.DataFrame(data = data_losses, columns=  ['epoch', 'tr_seg_loss', 'tr_disp_loss', 'tr_loss', 'val_seg_loss', 'val_disp_loss', 'val_loss'])
            if not os.path.exists(save_losses):
                with open (save_losses,'w',newline='') as df:
                    df_losses.to_csv(df,index=False)
            else:
                df_losses.to_csv(save_losses,mode='a',index=False, header= not os.path.exists(save_losses))

        writer.add_scalar('Val/seg_loss',avg_val_seg_loss,epoch+1)
        writer.add_scalar('Val/disparity_loss',avg_val_disp_loss,epoch+1)  
        writer.add_scalar('Val/total_loss',avg_val_loss,epoch+1) 
        writer.add_scalar('Val/mIoU',val_miou,epoch+1)
        writer.add_scalar('Val/RMSE',rmse_val,epoch+1) 
        writer.add_scalar('Val/AEPE',aepe_val,epoch+1) 
      
        writer.flush()
        writer.close()

        if early_stopping :
            early_stop=EarlyStopping()
            early_stop(avg_val_loss)
            if early_stop.early_stop :
                print('Early Stopping has been executed ')
                break

        """Save model state """       
        save_basename = roots.model_name + "_epoch_" + str(epoch + 1) + "_alpha_" + str(params.pareto_alpha) + ".pth"
        print('Saving checkpoint', os.path.join(roots.weights_dir, save_basename))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schedular_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, os.path.join(roots.weights_dir, save_basename))
                  
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def main(args):
    """Perform_training"""
    args['architecture'] = 'shared_decoder'
    #args['weighting_method'] = 'uw'
    args['weighting_method'] = 'dwa'
    config = config_training_setup(args)
    training_routine(config)

if __name__ == '__main__':
    """Get_Arguments_and setup_config_class"""

    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--training_starting_epoch", nargs="?", type=int)
    parser.add_argument("-nepochs", "--num_training_epochs", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-lr", "--learning_rate", nargs="?", type=float)
    parser.add_argument("-crop", "--crop_size", nargs="?", type=int)
    parser.add_argument("-weight_method", "--weighting_method", nargs= "?", type=str, help= 'type of weighting method')
    parser.add_argument("-architecture", "--architecture",nargs='?',type=str,help='Type of architecture')
    parser.add_argument("-backbone","--backbone", nargs="?",type= str,help='Backbone for Encoder',default='WideResNet38')

    main(vars(parser.parse_args()))
