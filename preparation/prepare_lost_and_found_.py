## use thisn script to generate binary mask for LostAndFound dataset
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
import sys
from natsort import natsorted


def mask_labelTrainIds(img,id_in,id_out):
    mask=np.ones(img.shape).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j ] == 255:
                mask[i,j]=id_in 
            elif img[i,j]==0:
                mask[i,j]=id_in
            elif img[i,j]==1:
                mask[i,j]=id_in
            else:
                mask[i,j]=id_out  
    return mask.astype(np.uint8)
def main():
    start = time.time()
    id_in=0
    id_out=254
    labelTrainIds=[]
    root="home/ravibhaivaghasiya/ravivaghasiya/datasets/LostAndFound/"
    split='train'
    save_dir='{}/gtCoarse/annotations/{}'.format(root,split)
    print("\nPrepare LostAndFound {} split for OoD training".format(split))
    for root, _, filenames in os.walk(os.path.join(root, 'gtCoarse', split)):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                filename_base = '_'.join(filename.split('_')[:-2])
                city = '_'.join(filename.split('_')[:-4])
                if '_'.join(filename.split('_')[-2:]) == 'gtCoarse_labelTrainIds.png':
                    labelTrainIds.append(os.path.join(root,filename))
    
    labelTrainIds=natsorted(labelTrainIds)
    
    num_masks = 0
    
    # Process each image
    
    print("Ground truth segmentation mask will be saved in:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created save directory:", save_dir)

    for i,img in enumerate(labelTrainIds):
        
        #open image from list of labelTrainIds
        image=np.array(Image.open(img).convert('L')).astype(np.uint8)
        
        #extract filename to save in save dir
        path,file=os.path.split(img)
        savename,ext=os.path.splitext(file)
        
        #generate binary mask from labelTrainIds with 0 and 254 value
        trainlabelmask=mask_labelTrainIds(image,id_in=id_in,id_out=id_out)
        
        # save each generated mask
        Image.fromarray(trainlabelmask).save(os.path.join(save_dir,savename+'_ood_mask.png'))
    
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


        
