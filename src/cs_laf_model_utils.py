import os
import sys

sys.path.insert(0,"/home/ravi/ravivaghasiya/meta-ood-master/ood_detection")
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from src.model.deepv3 import DeepWV3Plus, DeepR101V3PlusD_m1


gpu_ids =[1,0]

def load_network(model_name,num_classes,architecture, ckpt_path=None, train=False):
    network = None
    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)
    if model_name == "DeepLabV3+_WideResNet38":
        network = DeepWV3Plus(num_classes,architecture=architecture)
        network = nn.DataParallel(network,device_ids=[0,1])
    elif model_name =="DeepLabV3+_ResNet101":
        network = nn.DataParallel(DeepR101V3PlusD_m1(num_classes,criterion=None,architecture=architecture),device_ids=[0])
    else:
        print("\nModel is not known")
        exit()
    network = network.cuda(0)
    if ckpt_path is not None:
        if model_name =="DeepLabV3+_ResNet101":
            checkpoint=torch.load(ckpt_path,map_location=torch.device('cpu'))
            network.load_state_dict(checkpoint['state_dict'],strict=False)
        else :
            checkpoint=torch.load(ckpt_path,map_location=torch.device('cpu'))
            network.load_state_dict(checkpoint['state_dict'],strict=False)
        
    print('..... ok')
    return network , checkpoint

def prediction(net, image):
    image = image.cuda(0)
    with torch.no_grad():
        out_seg,out_depth = net(image)
    if isinstance(out_seg, tuple):
        out_seg = out_seg[0]
    out_seg = out_seg.data.cpu()
    out_seg = F.softmax(out_seg, 1)
    out_seg=out_seg.numpy()
    out_depth=out_depth.data.cpu()
    out_depth=out_depth.numpy()
    return out_seg,out_depth

class inference(object):

    def __init__(self, params,config, roots, loader,num_classes=None, init_net=True):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.model_name = roots.model_name
        self.architecture = config.architecture
        self.batch = 0
        self.batch_max = int(len(loader) / self.batch_size) + (len(loader) % self.batch_size > 0)
        self.loader = loader
        self.batchloader = iter(DataLoader(loader, batch_size=self.batch_size, shuffle=False))
        self.probs_root = os.path.join(roots.io_root, "probs")

        if self.epoch == 0:
            ckpt_path = roots.init_ckpt
            pattern = "baseline"            
            self.probs_load_dir = os.path.join(self.probs_root, pattern)
        else:
            pattern = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha)
            basename = self.model_name + "_" + pattern + ".pth"
            self.probs_load_dir = os.path.join(self.probs_root, pattern)
            ckpt_path = os.path.join(roots.weights_dir, basename)
        if init_net and num_classes is not None:
            basename = self.model_name + "_" + pattern + ".pth"
            ckpt_path = os.path.join(roots.weights_dir,basename)
            self.net,checkpoint = load_network(self.model_name, num_classes, ckpt_path=ckpt_path,architecture=self.architecture)
            self.net.eval()

    def probs_gt_load(self, i, load_dir=None):
        if load_dir is None:
            load_dir = self.probs_load_dir
        try:
            filename = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
            f_probs = h5py.File(filename, "r")
            probs_seg = np.asarray(f_probs['probabilities_seg'])
            probs_depth = np.asarray(f_probs['probabilities_depth'])
            gt_train = np.asarray(f_probs['gt_train_ids'])
            gt_depth = np.asarray(f_probs['gt_depth'])
            gt_label = np.asarray(f_probs['gt_label_ids'])
            probs_seg = np.squeeze(probs_seg)
            probs_depth = np.squeeze(probs_depth)
            gt_train = np.squeeze(gt_train)
            gt_depth=np.squeeze(gt_depth)
            gt_label = np.squeeze(gt_label)
            im_path = f_probs['image_path'][0].decode("utf8")
        except OSError:
            #print("No probs file for image %d, therefore run inference..." % i)
            probs_seg,probs_depth, gt_train,gt_depth, gt_label, im_path = self.prob_gt_calc(i)
        return probs_seg,probs_depth, gt_train, gt_depth,gt_label, im_path

    def probs_gt_save(self, i, save_dir=None):
        if save_dir is None:
            save_dir = self.probs_load_dir
        if not os.path.exists(save_dir):
            print("Create directory:", save_dir)
            os.makedirs(save_dir)
        probs_seg,probs_depth, gt_train,gt_depth, gt_label, im_path = self.prob_gt_calc(i)
        file_name = os.path.join(save_dir, "probs" + str(i) + ".hdf5")
        f = h5py.File(file_name, "w")
        f.create_dataset("probabilities_seg", data=probs_seg)
        f.create_dataset("probability_depth", data=probs_depth)
        f.create_dataset("gt_train_ids", data=gt_train)
        f.create_dataset("gt_depth", data=gt_depth)
        f.create_dataset("gt_label_ids", data=gt_label)
        f.create_dataset("image_path", data=[im_path.encode('utf8')])
        print("file stored:", file_name)
        f.close()

    def probs_gt_load_batch(self):
        assert self.batch_size > 1, "Please use batch size > 1 or use function 'probs_gt_load()' instead, bye bye..."
        x, y, z,d, im_paths = next(self.batchloader)
        probs_seg,probs_depth = prediction(self.net, x)
        gt_train = y.numpy()
        gt_label = z.numpy()
        gt_dmap= d.numpy()
        self.batch += 1
        print("\rBatch %d/%d processed" % (self.batch, self.batch_max))
        sys.stdout.flush()
        return probs_seg,probs_depth, gt_train,gt_dmap, gt_label, im_paths

    def prob_gt_calc(self, i):
        x, y,d = self.loader[i]
        
        probs_seg,probs_depth = prediction(self.net, x.unsqueeze_(0))
        probs_seg=np.squeeze(probs_seg)
        gt_train = y.numpy()
        gt_depth=d.numpy()
        try:
            gt_label = np.array(Image.open(self.loader.annotations[i]).convert('L'))
        except AttributeError:
            gt_label = np.zeros(gt_train.shape)
        im_path = self.loader.images[i]
        return probs_seg ,probs_depth, gt_train, gt_depth,gt_label, im_path


def probs_gt_load(i, load_dir):
    try:
        filepath = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
        f_probs = h5py.File(filepath, "r")
        probs_seg = np.asarray(f_probs['probabilities_seg'])
        probs_depth = np.asarray(f_probs['probability_depth'])
        gt_train = np.asarray(f_probs['gt_train_ids'])
        gt_depth = np.asarray(f_probs['gt_depth'])
        gt_label = np.asarray(f_probs['gt_label_ids'])
        probs_seg = np.squeeze(probs_seg)
        probs_depth = np.squeeze(probs_depth)
        gt_train = np.squeeze(gt_train)
        gt_depth = np.squeeze(gt_depth)
        gt_label = np.squeeze(gt_label)
        im_path = f_probs['image_path'][0].decode("utf8")
    except OSError:
        probs_seg,probs_depth, gt_train,gt_depth, gt_label, im_path = None, None, None, None,None
        print("No probs file, see src.cs_laf_model_utils")
        exit()
    return probs_seg  ,probs_depth, gt_train,gt_depth, gt_label, im_path
