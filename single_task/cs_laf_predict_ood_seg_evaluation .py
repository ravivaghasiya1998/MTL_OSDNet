import argparse
import os
import time
import sys
import numpy as np
import torch
import pickle
import torch
from tqdm import tqdm
from config_ood_seg import config_evaluation_setup
from src.imageaugmentations_ood_seg import Compose, Normalize, ToTensor
from src.cs_laf_predict_ood_seg_model_utils import inference
from scipy.stats import entropy
from src.metrics import StreamSegMetrics
from src.calc import calc_precision_recall, calc_sensitivity_specificity
from semantic_map import Sematic_Map

class eval_pixels(object):
    """
    Evaluate in vs. out separability on pixel-level
    """

    def __init__(self, params, roots, dataset):
        self.params = params
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.roots = roots
        self.dataset = dataset
        self.result_dir=os.path.join(self.roots.io_root, 'results')
        self.save_dir_data = os.path.join(self.roots.io_root, "results/entropy_counts_per_pixel")
        self.save_dir_plot = os.path.join(self.roots.io_root, "plots")
        self.save_dir_metrics=os.path.join(self.roots.io_root,"results/metrics")
        if self.epoch == 0:
            self.pattern = "baseline"
            self.save_path_data = os.path.join(self.save_dir_data, "baseline.p")
            self.save_metrics_data=os.path.join(self.save_dir_metrics, "baseline.p")
        else:
            self.pattern = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha)
            self.save_path_data = os.path.join(self.save_dir_data, self.pattern + ".p")
            self.save_metrics_data=os.path.join(self.save_dir_metrics, self.pattern + ".p")
            
    def counts(self, loader, num_bins=100, save_path=None, save_metrics_path=None,rewrite=False,Cityscapes=True):
        """
        Count the number in-distribution and out-distribution pixels
        and get the networks corresponding confidence scores
        :param loader: dataset loader for evaluation data
        :param num_bins: (int) number of bins for histogram construction
        :param save_path: (str) path where to save the counts data
        :param rewrite: (bool) whether to rewrite the data file if already exists
        """
        print("\nCounting in-distribution and out-distribution pixels")
        if save_path is None or save_metrics_path is None:
            save_path = self.save_path_data
            save_metrics_path=self.save_metrics_data
        if not os.path.exists(save_path) or rewrite:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                print("Create directory", save_dir)
                os.makedirs(save_dir)
        save_ori_sem_disp_ent=os.path.join(self.result_dir,'Epoch_'+str(self.epoch),'ori_seg_disp_ent')
        save_segmentation=os.path.join(self.result_dir,'Epoch_'+ str(self.epoch),'segmetation')
        save_ent=os.path.join(self.result_dir,'Epoch_'+str(self.epoch),'entropy_map')

        if not os.path.exists(save_ori_sem_disp_ent) :
            os.makedirs(save_ori_sem_disp_ent)
        if not os.path.exists(save_segmentation):
            os.makedirs(save_segmentation)
        if not os.path.exists(save_ent):
            os.makedirs(save_ent)
        print('all directories to save results are created')

        if not os.path.exists(save_metrics_path) or rewrite:
            save_metrics_dir=os.path.dirname(save_metrics_path)
            if not os.path.exists(save_metrics_dir):
                print("Create directory for metrics",save_metrics_dir)
                os.makedirs(save_metrics_dir)
        
            bins = np.linspace(start=0, stop=1, num=num_bins + 1)
            counts = {"in": np.zeros(num_bins, dtype="int64"), "out": np.zeros(num_bins, dtype="int64")}
            inf = inference(self.params, self.roots, loader, self.dataset.num_eval_classes)
            
            eval_metrics=StreamSegMetrics(n_classes=19)
            eval_metrics.reset()
            
            loader=tqdm(loader,leave=True,position=0)
            for i in range(len(loader)):                
                probs_seg,gt_train,gt_label, im_path = inf.probs_gt_load(i)
                eval_metrics.update(gt_train,probs_seg,train=False)
                # ## select class with max prob from 19 classes             
                max_index=np.argmax(probs_seg,axis=0)
                max_index=np.asarray(max_index).astype(np.uint8)

                # Segmentation Mask
                height,width=probs_seg.shape[1:]
                m,n=np.ogrid[:height,:width]
                probs_max=probs_seg[max_index,m,n]
                probs_max=np.array(probs_max)
                seg=Sematic_Map(max_index)
                # Entropy 
                ent = entropy(probs_seg, axis=0) / np.log(self.dataset.num_eval_classes)                             
                counts["in"] += np.histogram(ent[gt_train == self.dataset.train_id_in], bins=bins, density=False)[0]
                counts["out"] += np.histogram(ent[gt_train == self.dataset.train_id_out], bins=bins, density=False)[0]
                print("\rImages Processed: {}/{}".format(i + 1, len(loader)), end=' ')
                sys.stdout.flush()
            score=eval_metrics.get_results() 
            miou=score["Mean IoU"]  
            print(eval_metrics.to_str(score))
            print('Test mean IOU is {}'.format(miou))
            performance_metrics={'mIoU' : miou}
            torch.cuda.empty_cache()
            pickle.dump(counts, open(save_path, "wb"))
            pickle.dump(performance_metrics,open(save_metrics_path, "wb"))
        print("Counts data saved:", save_path,'and', "Performance metrics saved :",save_metrics_path)

    def oodd_metrics_pixel(self, datloader=None, load_path=None,load_metrics_path=None):
        """
        Calculate 3 OoD detection metrics, namely AUROC, FPR95, AUPRC
        :param datloader: dataset loader
        :param load_path: (str) path to counts data (run 'counts' first)
        :return: OoD detection metrics
        """
        if load_path is None or load_metrics_path is None:
            load_path = self.save_path_data
            load_metrics_path=self.save_metrics_data
        if not os.path.exists(load_path) or not os.path.exists(load_metrics_path):
            if datloader is None:
                print("Please, specify dataset loader")
                exit()
            self.counts(loader=datloader, save_path=load_path,save_metrics_path=load_metrics_path)
        data = pickle.load(open(load_path, "rb"))
        test_metrics=pickle.load(open(load_metrics_path,"rb"))
        fpr, tpr, _, auroc = calc_sensitivity_specificity(data, balance=True)
        fpr95 = fpr[(np.abs(tpr - 0.95)).argmin()]
        _, _, _, auprc = calc_precision_recall(data)
        if self.epoch == 0:
            print("\nOoDD Metrics - Epoch %d - Baseline" % self.epoch)
        else:
            print("\nOoDD Metrics - Epoch %d - Lambda %.2f" % (self.epoch, self.alpha))
        print("AUROC:", auroc)
        print("FPR95:", fpr95)
        print("AUPRC:", auprc)
        print ("mIoU:", test_metrics['mIoU'])
        return auroc, fpr95, auprc


def main(args):
    args['TRAINSET'] = 'Cityscapes+LAF'
    #args['VALSET'] = 'Cityscapes'
    args['VALSET'] = 'LostAndFound'
    #args['VALSET'] = 'SmallScale'
    args['MODEL'] = 'DeepLabV3+_WideResNet38'    
    config = config_evaluation_setup(args)
    if not args["pixel_eval"] and not args["segment_eval"]:
        args["pixel_eval"] = args["segment_eval"] = True

    transform = Compose([ToTensor(), Normalize(config.dataset.mean, config.dataset.std)])
    datloader = config.dataset(root=config.roots.eval_dataset_root,split='test', transform=transform)
   
    """Perform evaluation"""
    print("\nEVALUATE MODEL: ", config.roots.model_name)
    if args["pixel_eval"]:
        print("\nPIXEL-LEVEL EVALUATION")
        eval_pixels(config.params, config.roots, config.dataset).oodd_metrics_pixel(datloader=datloader)

    start = time.time()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-val", "--VALSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--val_epoch", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-pixel", "--pixel_eval", action='store_true')
    parser.add_argument("-segment", "--segment_eval", action='store_true')
    main(vars(parser.parse_args()))
