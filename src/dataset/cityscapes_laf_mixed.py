import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys
import cv2
import numpy as np
sys.path.insert(0,"/home/ravi/ravivaghasiya/cs_laf_predict/ood_seg_disp/src")
from dataset.laf_ood_proxy_predict import LostAndFound
from dataset.cityscapes import Cityscapes
from imageaugmentations import ToTensor,Compose,RandomCrop,RandomHorizontalFlip,Normalize
from sklearn.utils import shuffle as shuffled
random.seed(42)

class CityscapesLAFMix(Dataset):

    def __init__(self, split='train', transform=None,
                 cs_root="/home/ravi/ravivaghasiya/dataset/cityscapes",
                 laf_root="/home/ravi/ravivaghasiya/dataset/LostAndFound",
                 subsampling_factor=0.1, cs_split=None, laf_split=None,shuffle=True):
        transform = Compose([RandomHorizontalFlip(), RandomCrop(480), ToTensor(),
                         Normalize(Cityscapes.mean, Cityscapes.std)])
        self.transform = transform
        if cs_split is None or laf_split is None:
            self.cs_split = split
            self.laf_split = split
        else:
            self.cs_split = cs_split
            self.laf_split = laf_split

        self.cs = Cityscapes(root=cs_root, split='train')
        self.lostandfound =LostAndFound(root=laf_root, split=self.laf_split)
        self.images = self.cs.images + self.lostandfound.images
        self.labelTrainIds=self.cs.labelTrainIds+self.lostandfound.labelTrainids
        self.disparity=self.cs.disparity+self.lostandfound.disparity
        self.ids= [0]*len(self.cs.images) +[1] * len(self.lostandfound.images)
        self.train_id_out = self.lostandfound.train_id_out
        self.ood_ind=254
        self.num_classes = self.cs.num_train_ids
        self.mean = self.cs.mean
        self.std = self.cs.std
        self.void_ind = self.cs.ignore_in_eval_ids
        
        if shuffle:
            self.images,self.labelTrainIds,self.disparity,self.ids=shuffled(self.images,self.labelTrainIds,self.disparity,self.ids)
            
    def __getitem__(self, i):
        """Return raw image, ground truth in PIL format and absolute path of raw image as string"""
        image = np.asarray(Image.open(self.images[i]).convert('RGB')).astype(np.uint8)  
        labelTrainId = np.asarray(Image.open(self.labelTrainIds[i])).astype(np.uint8)
        disparity_map=cv2.imread(self.disparity[i],cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity_map[disparity_map > 0]=(disparity_map[disparity_map > 0] - 1. )/256.
        ids=self.ids[i]

        if self.transform is not None:
            image, labelTrainId,disparity_map = self.transform(image, labelTrainId,disparity_map)
        return image,labelTrainId,disparity_map,ids

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.cs)
        fmt_str += 'LAF Split: %s\n' % self.laf_split
        fmt_str += '----Number of images: %d\n' % len(self.lostandfound.images)
        return fmt_str.strip()


