**<h1>Detecting the Unexpected: A Safety Enabled Multi-Task Approach Towards Unknown Object-Segmentation and Depth Estimation</h1>**

 **Abstract:** This work addresses the issue of detecting unexpected road obstacles on the road and improving visual scene understanding using Multi-Task Learning (MTL) on monocular images. The latest semantic segmentation Convolutional Neural Networks (CNNs) are limited to identifying a fixed pre-defined set of objects. However, in real-time applications like autonomous driving, these CNNs face challenges in detecting objects that do not belong to their semantic space. These objects are termed as Out-of-Distribution (OoD) in this work. Vision-based detection of OoD instances is crucial for safety-relevant applications. To this end, we propose a novel approach using MTL to detect OoD in semantic segmentation and disparity estimation. We propose a novel MTL network referred to as ”OSDNet”. To detect the OoD objects, we follow the identical approach proposed by [Robin et al.](https://arxiv.org/pdf/2012.06575.pdf). Our MTL baseline is defined by training the proposed network using [Cityscapes](https://www.cityscapes-dataset.com/) as "in-distribution" data and [LostAndFound](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) as OoD data. To improve the accuracy of each task, we develop a semi-automatic relabeling technique to obtain semantic labels for the ”known” objects in the LostAndFound dataset, which originally lacked such labels. Compared to MTL baseline, training the same network with the relabeled data as the OoD proxy resulted in an increment of 7.2 % in mean Intersection over Union (mIoU) with a score of 80.46 %. This approach reduced the detection error measured by False Positive Rate at 95 % (FPR95) True Positive Rate (TPR) by 10% from 0.16 to 0.06. 
 
---

</br>

## Requirements ##

This code is built and tested with **[Python 3.8.10](https://www.python.org/downloads/release/python-3810/)** and **[CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)**. The required dependencies were installed via **pip 20.0.2**. Please install the required libraries via:

```
  pip install requirements.txt
````
``` 
├── cityscapes
│   ├── leftImg8bit
│   ├── gtFine
│   ├── disparity
│   └── weights
├── LostAndFound
│   ├── leftImg8bit
│   ├── gtFine
│   └── disparity
└── Small_obstacles
    ├── depth
    ├── images
    └── labels

``` 
**Data preprocessing:** The Dataloader used in this repo assumes that the *labelTrainId* images for Cityscapes dataset are already generated according to [official Cityscapes script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).

The script in ` preparation/prepare_lost_and_found.py` can be used to generate binary mask of LostAndFound dataset serving as OoD proxy data.

Regarding the semantic relabeling of the LostAndFound dataset, please use the DeepLabv3+ network and pre-trained weights from https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet and run the inference on LostAndFound left RGB images. After getting predicted semantic labels, the script in ` preparation/relabel_predicted_lost_and_found.py ` can be used to manually assign the unique *labelTrainId* to OoD pixels  which are classified as one of the 19 classes of Cityscapes. Please keep in mind to change the file names and directory path according to your setup. One can download the relabeled LostAndFound dataset directly [here](https://www.dropbox.com/scl/fo/8bqjlqrrgqzkb4ha3zhwc/h?dl=0&rlkey=yavgb62gwhxo233qllpna5ffa)

## Implementation ##
Modify the settings in ` config.py `. All the files will be saved in directory defined via `io_root`. Don't forget to change the root directory of all datasets. Then to train the network, please run:

```
python3 cs_laf_predict_multitask_training .py
````

If no command-line arguments are provided to select particular versions of network as well as particular task-weighting methods (i.e. `args['architecture']` and ` args['weighting_method']`) then the default areguments in `config.py` and `cs_laf_predict_multitask_training .py` will be applied.

To evaluate the network for particular epoch, change the settings in `config.py' as well as select arguments for dataset on which you want to evalate the model. Then run:

```
python3 cs_laf_evaluation .py
````

If no command-line arguments for `args['architecture']` and `args['VALSET']` selected then defaults options in `cs_laf_evaluation .py` will be used.

** Reproducing the results **

The weights after training the network using Cityscapes and relabeled LostAndFound datasets can be downloaded [here]() to reproduce the same results for each individual tasks.

To perform the training for Single-Task Learning approach, the scripts provided in `single_task`can be used and updated according to task that needs to be performed.
