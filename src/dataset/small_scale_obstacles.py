import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
import sys
sys.path.insert(0,"/home/ravi/ravivaghasiya/cs_laf_predict/ood_seg_disp/src")
from src.dataset.cityscapes import Cityscapes
import cv2
class SmallScale(Dataset):

    train_id_in = 1
    train_id_out = 2
    cs = Cityscapes()
    mean = cs.mean
    std = cs.std
    num_eval_classes = cs.num_train_ids

    def __init__(self, split='test', root="/home/ravi/ravivaghasiya/dataset/Small_obstacles", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split      # ['test', 'train']
        self.images = []        # list of all raw input images
        self.targets = []       # list of all ground truth TrainIds images
        self.disparity=[]       # list of all disparity maps

        for root, _, filenames in os.walk(os.path.join(root, 'images', self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    self.images.append(os.path.join(root, filename))
                    street=(os.path.split(os.path.join(root,filename))[0]).split('/')[-1]
                    target_root = os.path.join(self.root, 'labels', self.split)
                    self.targets.append(os.path.join(target_root,street, filename ))#target_root,city,filename_base.....
                    disparity_root = os.path.join(self.root,'depth',self.split)
                    self.disparity.append(os.path.join(disparity_root, street, filename))
        self.images=natsorted(self.images)
        self.targets=natsorted(self.targets)
        self.disparity=natsorted(self.disparity)##natural sorting of images

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
        if self.split == 'test':
            image = np.asarray(Image.open(self.images[i]).convert('RGB'))  
            labelTrainId=np.asarray(Image.open(self.targets[i])).astype(np.uint8)
            disparity_map=cv2.imread(self.disparity[i],cv2.IMREAD_UNCHANGED).astype(np.float32)
            
            if self.transform is not None:
                
                image, labelTrainId,disparity_map = self.transform(image, labelTrainId,disparity_map)
        return image,labelTrainId,disparity_map

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'SmallScale Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

