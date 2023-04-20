## load label containing predicted class index from DeeplabV3+ and 
# load original laf label and replace ood id in predicted label and assign 25 to roi in predicted label

import time
from matplotlib import pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
import sys
from natsort import natsorted

def labelIds_color_roi(ori_lbl_c,roi_c):
    labelIds_c_roi = ori_lbl_c * roi_c        
    return labelIds_c_roi
def relabel_labelTrainIds(ori_lbl,pred_lbl,roi):
   
    for i in range(ori_lbl.shape[0]):
        for j in range(ori_lbl.shape[1]):
            if roi[i,j ] == 0:#roi is 0
                pred_lbl[i,j]= 255 #void_class as in cityscape to ignore roi
            elif ori_lbl[i,j]== 2: ##ood id in original laf label
                pred_lbl[i,j] = 254
         
    return pred_lbl.astype(np.uint8)
def main():
    start = time.time()
    labelTrainIds=[]
    original_lbl =  []
    labelIds_color = []
    root1="/home/ravibhaivaghasiya/ravivaghasiya/dataset/LostAndFound_pred/"
    split='train'
    save_root = "/home/ravibhaivaghasiya/ravivaghasiya/dataset"
    save_dir= os.path.join(save_root,'LAF_with_ROI',split)
    print("\nPrepare LostAndFound {} split for OoD training".format(split))
    for root, _, filenames in os.walk(os.path.join(root1, split)):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                filename_base = '_'.join(filename.split('_')[:-4])
                city = '_'.join(filename.split('_')[:-6])
                if '_'.join(filename.split('_')[-2:]) == 'predict_labelTrainIds.png':
                    labelTrainIds.append(os.path.join(root,filename))#target_root,city,filename_base.....
                    original_lbl.append(os.path.join(save_root,'LostAndFound','gtCoarse','train_ori',city,filename_base+'_gtCoarse_labelTrainIds.png'))
                    labelIds_color.append (os.path.join(root1,split+'_','color',city,filename_base+'_leftImg8bit_gtCoarse_predict_color.png'))
    
    labelTrainIds=natsorted(labelTrainIds)
    original_lbl = natsorted(original_lbl)
    labelIds_color =natsorted(labelIds_color)
    roipath = '/home/ravibhaivaghasiya/ravivaghasiya/dataset/roi_image.png'
    roi = np.array(Image.open(roipath).convert('L'))
    roi_color = np.array(Image.open(roipath).convert('RGB'))
    num_masks = 0
    
    # Process each image
    
    print("Ground truth segmentation mask will be saved in:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created save directory:", save_dir)

    for i,(pred_l,ori_lbl,pred_color) in enumerate(zip(labelTrainIds,original_lbl,labelIds_color)):
 
        pre_lbl=np.array(Image.open(pred_l))
        o_lbl = np.array (Image.open(ori_lbl))
        pre_color = np.array(Image.open(pred_color).convert('RGB'))
        #extract filename to save in save dir
        path,file=os.path.split(pred_l)
        savename='_'.join(file.split('_')[:-4])

        
        #generate binary mask from labelTrainIds with 0 and 254 value
        TrainIds_roi=relabel_labelTrainIds(o_lbl,pre_lbl,roi)        
        # save each generated mask
        Image.fromarray(TrainIds_roi).save(os.path.join(save_dir,'TrainIds',savename+'_gtCoarse_predict_labelTrainIds.png'))
       
        num_masks+=1
        print("\rImages Processed: {}/{}".format(i + 1, len(labelTrainIds)), end=' ')
        sys.stdout.flush()
        
    # Print summary
    print("\nNumber of created segmentation masks of labelTrainIds are %d :" % num_masks)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
#img=np.array(Image.open(cpath).convert('L')).astype(np.uint8)
#print(img.dtype)
# plt.imshow(img)
# plt.show()

#print(np.unique(mask))    
# plt.imshow(mask)
# plt.show()    
#plt.imsave('C:/Users/TUQC9OT/Desktop/datasets/lost_and_found/gtCoarse/train/01_Hanns_Klemm_Str_45_000000_000270_gtCoarse_labelIds_binary.png',mask)  
#Image.fromarray(mask).convert('L').save('C:/Users/TUQC9OT/Desktop/datasets/lost_and_found/gtCoarse/train/01_Hanns_Klemm_Str_45_000000_000270_gtCoarse_labelIds_binary.png')

        
