# Melanoma-Detection-Assignment
# Project Name
To build a multiclass classification model using a custom convolutional neural network in TensorFlow. 


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)


## General Information
To build a CNN based model which can accurately detect melanoma. 
Melanoma is a type of cancer that can be deadly if not detected early.
It accounts for 75% of skin cancer deaths. A solution that can evaluate 
images and alert dermatologists about the presence of melanoma has the 
potential to reduce a lot of manual effort needed in diagnosis.
The dataset consists of 2357 images of malignant and benign oncological diseases, 
which were formed from the International Skin Imaging Collaboration (ISIC). 
All images were sorted according to the classification taken with ISIC, 
and all subsets were divided into the same number of images, with the exception of melanomas and moles, 
whose images are slightly dominant.


The data set contains the following diseases:

Actinic keratosis
Basal cell carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented benign keratosis
Seborrheic keratosis
Squamous cell carcinoma
Vascular lesion


## Conclusions

The model seems to be overfitting ,as we see the difference in loss functions in training & test in between the 10-14th epoch.
We dont see much improvement in accuracy but we can see the overfitting problem has been solved due to data augmentation.
It seems Seborrheic keratosis has least number of samples
Basal cell carcinoma , Melanoma and pigmented benign keratosis have proprtionate number of class

## Technologies Used
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from glob import glob
