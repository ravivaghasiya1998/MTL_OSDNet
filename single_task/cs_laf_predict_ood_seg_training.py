import argparse
import pickle
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchsummary import summary
from config_ood_seg import config_training_setup
from src.metrics import StreamSegMetrics
from src.imageaugmentations_ood_seg import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.cs_laf_predict_ood_seg_model_utils  import load_network
from src.early_stopping import EarlyStopping
from torch.utils.data import DataLoader,random_split 
from torch.utils.tensorboard import SummaryWriter

def cross_entropy(logits, targets):
    """
    cross entropy loss with one/all hot encoded targets -> logits.size()=targets.size()
    :param logits: torch tensor with logits obtained from network forward pass
    :param targets: torch tensor one/all hot encoded
    :return: computed loss
    """

    neg_log_like = - 1.0 * F.log_softmax(logits, 1)
    L = torch.mul(targets.float(), neg_log_like)
    L = L.mean()
    return L


def encode_target(target, pareto_alpha, num_classes, ignore_train_ind, ood_ind=254):
    """
    encode target tensor with all hot encoding for OoD samples
    :param target: torch tensor
    :param pareto_alpha: OoD loss weight
    :param num_classes: number of classes in original task
    :param ignore_train_ind: void class in original task
    :param ood_ind: class label corresponding to OoD class
    :return: one/all hot encoded torch tensor
    """

    npy = target.numpy()
    npz = npy.copy()
    npy[np.isin(npy, ood_ind)] = num_classes
    npy[np.isin(npy, ignore_train_ind)] = num_classes + 1
    enc = np.eye(num_classes + 2)[npy][..., :-2]  # one hot encoding with last 2 axis cutoff
    enc[(npy == num_classes)] = np.full(num_classes, pareto_alpha / num_classes)  # set all hot encoded vector
    enc[(enc == 1)] = 1 - pareto_alpha  # convex combination between in and out distribution samples
    enc[np.isin(npz, ignore_train_ind)] = np.zeros(num_classes)
    enc = torch.from_numpy(enc)
    enc = enc.permute(0, 3, 1, 2).contiguous()
    return enc

def training_routine(config,early_stopping=True):
    """Start OoD Training"""
    print("START OOD TRAINING")
    params = config.params
    alpha=params.pareto_alpha
    roots = config.roots
    dataset = config.dataset()
    print("Pareto alpha:", params.pareto_alpha)
    result_dir=os.path.join(roots.io_root, 'results')
    save_dir_data = os.path.join(result_dir,'entropy_counts_per_pixel')
    save_dir_metrics=os.path.join(result_dir,'metrics')
    start_epoch = params.training_starting_epoch
    epochs = params.num_training_epochs
    start = time.time()


    """Initialize model"""
    if start_epoch == 0:
        network, checkpoint = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt)#, train=True)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network, checkpoint = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename))#, train=True)
    #print(summary(network,(3,480,480)))
    optimizer = optim.Adam(network.parameters(), lr=params.learning_rate)
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.9,patience=3,threshold=1,threshold_mode='abs',verbose=True)
    if start_epoch != 0:
        start_epoch=checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['schedular_state_dict'])
        print('\n current lr is', optimizer.param_groups[0]['lr'])


    transform = Compose([RandomHorizontalFlip(), RandomCrop(params.crop_size), ToTensor(),
                         Normalize(dataset.mean, dataset.std)])

    trainloader = config.dataset('train', transform,roots.cs_root, roots.laf_root,shuffle=True)
    train_loader,val_loader= random_split(trainloader,lengths=[int(0.9*len(trainloader)),(len(trainloader)-int(0.9*len(trainloader)))],generator=torch.Generator().manual_seed(42))
    train_data=DataLoader(train_loader,batch_size=params.batch_size,shuffle=True,drop_last=True)
    val_data=DataLoader(val_loader,batch_size=8,shuffle=True,drop_last=True)

    save_summary=os.path.join(roots.io_root,'summary')
    writer=SummaryWriter(save_summary,flush_secs=30)
    if not os.path.exists(save_summary):
        os.makedirs(save_summary)
        print('\n Summary save path directory is created at',str(save_summary))

    print('No. of train batches is {} and No. of Validation Batches is {}'.format(len(train_data),len(val_data)))
    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""
        """Perform one epoch of training"""
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
        train_seg_loss=0.0
        loss=0.0
        train_data=tqdm(train_data,position=0,leave=True)
        network.train()
        train_metrics=StreamSegMetrics(n_classes=dataset.num_classes)
        train_metrics.reset()
        for i,(x, target) in enumerate(train_data):
            optimizer.zero_grad()
            logits= network(x.cuda(1))
          
            train_metrics.update(label_trues=target,label_preds=logits,train=True)
            y = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
                              ignore_train_ind=dataset.void_ind, ood_ind=dataset.ood_ind).cuda(1)
            
            seg_loss = cross_entropy(logits, y)
            loss=seg_loss
            loss.backward()
            optimizer.step()
            train_seg_loss += seg_loss.item()
            train_data.set_description('Train : Epoch %i Loss : %.8f'%(epoch+1,loss))

            train_data.refresh()
        score=train_metrics.get_results() 
        miou=score["Mean IoU"] 
        scheduler.step(miou)
        lr=np.array([x['lr'] for x in optimizer.param_groups ])
        print('\n New lr is {}'.format(lr))
        
        print(' *'*10,' Training Results' , '*'*10) 
        print('\n mIoU_Train is {:f}'.format(miou))
        avg_train_seg_loss=train_seg_loss / len(train_data)
        writer.add_scalar('Train/seg_loss',avg_train_seg_loss,epoch+1)
        writer.add_scalar('Train/mIoU',miou,epoch+1)
        writer.add_scalar('Train / learning_rate',lr,epoch+1)        
        network.eval()
        val_loss=0.0
        val_seg_loss=0.0
        val_data=tqdm(val_data,position=0,leave=True)
        with torch.no_grad():
            val_metrics=StreamSegMetrics(n_classes=dataset.num_classes)
            val_metrics.reset()
            for i,(x,target) in enumerate(val_data):
                logits= network(x.cuda(1))
                val_metrics.update(label_trues=target,label_preds=logits,train=True)
                y = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
                                ignore_train_ind=dataset.void_ind, ood_ind=dataset.ood_ind).cuda(1) 
                seg_loss = cross_entropy(logits, y)
                loss=seg_loss.item()
                val_data.set_description('Validation : Epoch %i Loss : %.8f'%(epoch+1,loss))
                val_data.refresh()
                val_loss += loss
                val_seg_loss += seg_loss.item()
                
            score=val_metrics.get_results()
            vali_miou=score["Mean IoU"]   
            print('*'*10,' Validation Results' , '*'*10) 
            print('\n mIoU_Val is {:f}'.format(vali_miou)) 
        performance_metrics={'mIoU_train' : miou,'mIoU_val':vali_miou}
        pickle.dump(performance_metrics,open(save_metrics_data, "wb"))
        print("\n Performance metrics saved :",save_metrics_data)

        avg_val_loss=val_loss / len(val_data)
        avg_val_seg_loss=val_seg_loss / len(val_data)
        writer.add_scalar('Val/seg_loss',avg_val_seg_loss,epoch+1)
        writer.add_scalar('Val/mIoU',vali_miou,epoch+1)
        
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
        torch.cuda.empty_cache()
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def main(args):
    """Perform_training"""
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

    main(vars(parser.parse_args()))
