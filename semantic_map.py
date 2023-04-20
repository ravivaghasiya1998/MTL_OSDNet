import cv2
import numpy as np
from PIL import Image
from collections import namedtuple

# Label from Cityscapes

Label = namedtuple( 'Label' , ['name', 'id', 'trainId', 'color', ] )

id_to_trainid = [
    #       name                     id    trainId      color
    Label(  'unlabeled'            ,  0 ,      255 ,  (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 ,  (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 ,  (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 ,  (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 ,  (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 ,  (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 ,  ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 ,  (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 ,  (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 ,  (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 ,  (230,150,140) ),
    Label(  'building'             , 11 ,        2 ,  ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 ,  (102,102,156) ),
    Label(  'fence'                , 13 ,        4 ,  (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 ,  (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 ,  (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 ,  (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 ,  (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 ,  (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 ,  (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 ,  (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 ,  (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 ,  (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 ,  ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 ,  (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 ,  (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 ,  (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 ,  (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 ,  (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 ,  (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 ,  (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 ,  (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 ,  (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 ,  (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 ,  (  0,  0,142) )

]

# Color Map
C_map = []

for label_info in id_to_trainid:
    if label_info.trainId != 255:
        C_map.append(label_info.color)
    
C_map = np.array(C_map) 

def Sematic_Map(seg_img):
    roi_path='/home/ravi/ravivaghasiya/cs_laf_predict/ood_seg_disp/roi_image.png'
    roi=np.array(Image.open(roi_path).convert('RGB')).astype(np.uint8)
    
    pred_imgs_map = [C_map[p] for p in seg_img]
    pred_imgs_map = np.array(pred_imgs_map)
    #show = False
    # if show:
    #     fig, ax = plt.subplots(figsize=(10, 9))
    #     ax.imshow(pred_imgs_map) 
        
    return (pred_imgs_map).astype(np.uint8) * roi       
        

