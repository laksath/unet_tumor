#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, os.path
import re
import pandas as pd
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
import imageio
from imageio import imread
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil
import time

data=[([],[]),([],[]),([],[])]
data_url=[([],[]),([],[]),([],[])]

count_temp=0

directory="../resized/"
subdir=["benign/","malignant/","normal/"]


# In[2]:


def LoadData(imgPath,subd,ix,images,masked,images_url,masked_url,check):
    global count_temp
    count_temp+=1
    
    if(count_temp>3):
        for i in range(3):
            data[i][0].clear()
            data[i][1].clear()
        count_temp=1
        
    imgPath=imgPath+subd
    imgNames = os.listdir(imgPath)

    for item in imgNames:
        img = plt.imread(imgPath + item)
        if item.endswith(')_mask resized_greyscale.jpg'):
            masked.append(img)
        elif item.endswith(') resized.jpg'):
            # print(item)
            images.append(img)
        elif item.endswith(' resized_greyscale.jpg'):
            masked[-1]=np.add(masked[-1], img)

def LoadData_Info():
    for i in range(3):
        LoadData(directory,subdir[i],i,data[i][0],data[i][1],data_url[i][0],data_url[i][1],0)


# In[3]:


LoadData_Info()


# In[4]:


# benign image ; benign mask


# In[14]:


dir = 'benign_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'benign_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[0][0])):
    im = Image.fromarray(data[0][0][i])
    im.save("benign_image/"+str(i)+".jpeg")

for i in range(len(data[0][0])):
    im = Image.fromarray(data[0][1][i])
    im.save("benign_mask/"+str(i)+".jpeg")


# In[15]:


#malignant_image, malignant_mask


# In[19]:


dir = 'malignant_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'malignant_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[1][0])):
    im = Image.fromarray(data[1][0][i])
    im.save("malignant_image/"+str(i)+".jpeg")

for i in range(len(data[1][0])):
    im = Image.fromarray(data[1][1][i])
    im.save("malignant_mask/"+str(i)+".jpeg")


# In[20]:


# normal_image, normal_mask


# In[21]:


dir = 'normal_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'normal_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[2][0])):
    im = Image.fromarray(data[2][0][i])
    im.save("normal_image/"+str(i)+".jpeg")

for i in range(len(data[2][0])):
    im = Image.fromarray(data[2][1][i])
    im.save("normal_mask/"+str(i)+".jpeg")

