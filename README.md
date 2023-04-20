**<h1>Detecting the Unexpected: A Safety Enabled Multi-Task Approach Towards Unknown Object-Segmentation and Depth Estimation</h1>**

 **Abstract:** This work addresses the issue of detecting unexpected road obstacles on the road and improving visual scene understanding using Multi-Task Learning (MTL) on monocular images. The latest semantic segmentation Convolutional Neural Networks (CNNs) are limited to identifying a fixed pre-defined set of objects. However, in real-time applications like autonomous driving, these CNNs face challenges in detecting objects that do not belong to their semantic space. These objects are termed as Out-of-Distribution (OoD) in this work. Vision-based detection of OoD instances is crucial for safety-relevant applications. To this end, we propose a novel approach using MTL to detect OoD in semantic segmentation and disparity estimation. We propose a novel MTL network referred to as ”OSDNet”. To detect the OoD objects, we follow the identical approach proposed by [Robin et al.](https://arxiv.org/pdf/2012.06575.pdf). Our MTL baseline is defined by training the proposed network using [Cityscapes](https://www.cityscapes-dataset.com/) as "in-distribution" data and [LostAndFound](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) as OoD data. To improve the accuracy of each task, we develop a semi-automatic relabeling technique to obtain semantic labels for the ”known” objects in the LostAndFound dataset, which originally lacked such labels. Compared to MTL baseline, training the same network with the relabeled data as the OoD proxy resulted in an increment of 7.2 % in mean Intersection over Union (mIoU) with a score of 80.46 %. This approach reduced the detection error measured by False Positive Rate at 95 % (FPR95) True Positive Rate (TPR) by 62.5 % from 0.16 to 0.06, and the disparity error by 5 %. 

</br>

Download relabeled LostAndFound dataset [here](https://www.dropbox.com/scl/fo/8bqjlqrrgqzkb4ha3zhwc/h?dl=0&rlkey=yavgb62gwhxo233qllpna5ffa)

---

</br>

## Requirements ##

This code is built and tested with **[Python 3.8.10](https://www.python.org/downloads/release/python-3810/)** and **[CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)**. The required dependencies were installed via **pip 20.0.2**. Please install the required libraries via:

'''
pip install requirements.txt
'''
