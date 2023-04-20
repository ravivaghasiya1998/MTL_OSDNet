import os
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple
from typing import Callable, Optional
import sys
import numpy as np
sys.path.insert(0,"/home/ravi/ravivaghasiya/cs_laf_predict/ood_seg_disp")
from src.dataset.cityscapes_ood_seg import Cityscapes
from natsort import natsorted
import random
from sklearn.utils import shuffle as shuffled
random.seed(42)


class LostAndFound(Dataset):

    LostAndFoundClass = namedtuple('LostAndFoundClass', ['name', 'id', 'train_id', 'category_name',
                                                         'category_id', 'color'])

    labels = [
        LostAndFoundClass('unlabeled', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('ego vehicle', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('rectification border', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('out of roi', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('background', 0, 255, 'Counter hypotheses', 1, (0, 0, 0)),
        LostAndFoundClass('free', 1, 1, 'Counter hypotheses', 1, (128, 64, 128)),
        LostAndFoundClass('Crate (black)', 2, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - stacked)', 3, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - upright)', 4, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray)', 5, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - stacked) ', 6, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - upright)', 7, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Bumper', 8, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 1', 9, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue)', 10, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue - small)', 11, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green)', 12, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green - small)', 13, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Exhaust Pipe', 14, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Headlight', 15, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Euro Pallet', 16, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon', 17, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (large)', 18, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (white)', 19, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Rearview mirror', 20, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Tire', 21, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Ball', 22, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bicycle', 23, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (black)', 24, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (white)', 25, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Kid dummy', 26, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby car (gray)', 27, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (red)', 28, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (yellow)', 29, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 2', 30, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Marker Pole (lying)', 31, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Plastic bag (bloated)', 32, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Post (red - lying)', 33, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Post Stand', 34, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Styrofoam', 35, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Timber (small)', 36, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Timber (squared)', 37, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wheel Cap', 38, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wood (thin)', 39, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Kid (walking)', 40, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (on a bobby car)', 41, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (small bobby)', 42, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (crawling)', 43, 2, 'Humans', 6, (0, 0, 142)),
    ]

    train_id_in = 1
    train_id_out = 2
    cs = Cityscapes(split='train')
    mean = cs.mean
    std = cs.std
    num_eval_classes = cs.num_train_ids

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, shuffle=True)-> None:
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split      # ['test', 'train']
        self.images = []        # list of all raw input images
        self.labelTrainids=[]   # list of all labelTrainIDs for ooD dataset
        self.labelTrainIds_test=[]
        self.targets = []       # list of all ground truth TrainIds images and disparity maps
        self.annotations = []   # list of all ground truth LabelIds images
        

        for root, _, filenames in os.walk(os.path.join(root, 'leftImg8bit',self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    city = '_'.join(filename.split('_')[:-3])

                    ##get path of cropped images leftImg8bit from directory
                    self.images.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    ##get path of labelTrainids images from directory
                    labeltrainids_root = os.path.join(self.root, 'gtCoarse', self.split)
                    self.labelTrainids.append(os.path.join(labeltrainids_root,city, filename_base + '_gtCoarse_predict_labelTrainIds.png'))#target_root,city,filename_base.....
                    labbeltest=os.path.join(self.root,'gtCoarse',self.split,city,filename_base +'_gtCoarse_labelTrainIds.png')
                    self.labelTrainIds_test.append(labbeltest)
                    
                    annotation_root = os.path.join(self.root, 'gtCoarse', self.split)
                    self.annotations.append(os.path.join(annotation_root, city, filename_base + '_gtCoarse_labelTrainIds.png'))
        self.images=natsorted(self.images)
        self.labelTrainids=natsorted(self.labelTrainids)
        self.labelTrainIds_test=natsorted(self.labelTrainIds_test)
        """
        shuffle data and subsample
        """
        if shuffle :
            self.images,self.labelTrainids=shuffled(self.images,self.labelTrainids)
        
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
   
        if self.split=='test':
            self.images=natsorted(self.images)
            self.labelTrainids=natsorted(self.labelTrainIds_test)            
            image = np.asarray(Image.open(self.images[i])).astype(np.uint8)    
            labelTrainId=np.asarray(Image.open(self.labelTrainIds_test[i])).astype(np.uint8)
            if self.transform is not None:
                image, labelTrainId= self.transform(image, labelTrainId)
        else:
            image=self.images[i]
            labelTrainId=self.labelTrainids[i]

        return image,labelTrainId

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()
