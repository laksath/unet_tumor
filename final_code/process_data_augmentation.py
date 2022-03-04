#!/usr/bin/env python
# coding: utf-8

# In[1]:


train_test_split_=0.7
train_valid_split_=0.75 #after augmentation


# In[2]:


import os, os.path
import re
import pandas as pd
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
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
import albumentations as A


# In[3]:


########################################################################################################################
####################################         AUGMENTATION            ###################################################
########################################################################################################################


# In[4]:


def light_augmentation(image,mask):
    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5)]
    )

    augmented = aug(image=image, mask=mask)

    image_light = augmented['image']
    mask_light = augmented['mask']
    
    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5)]
    )

    augmented = aug(image=image, mask=mask)

    image_light = augmented['image']
    mask_light = augmented['mask']
    
    return (image_light,mask_light)


# In[5]:


def medium_augmentation(image,mask):
    aug = A.Compose([
        A.RandomSizedCrop(min_max_height=(100, 101), height=256, width=256, p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.8)])

    augmented = aug(image=image, mask=mask)

    image_medium = augmented['image']
    mask_medium = augmented['mask']
    
    return (image_medium,mask_medium)


# In[6]:


def heavy_augmentation(image,mask):
    aug = A.Compose([
#         A.RandomSizedCrop(min_max_height=(100, 101), height=256, width=256, p=0.5),
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)                  
            ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),    
        A.RandomGamma(p=0.8)])

    augmented = aug(image=image, mask=mask)

    image_heavy = augmented['image']
    mask_heavy = augmented['mask']
    
    return (image_heavy,mask_heavy)


# In[7]:


########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################


# In[8]:


directory="before_augment/"
subdir=["benign_image/","benign_mask/","malignant_image/","malignant_mask/","normal_image/","normal_mask/"]
dataset=['dataset/','train/','test/','validation/']


# In[9]:


########################################################################################################################
####################################         StoreData           #######################################################
########################################################################################################################


# In[10]:


def StoreData(directory,subdir):
    for data in dataset:
        if data=='dataset/':
            if os.path.exists(data):
                shutil.rmtree(data)
            os.makedirs(data)
        else:
            if os.path.exists('dataset/'+data):
                shutil.rmtree('dataset/'+data)
            os.makedirs('dataset/'+data)
            for sub in subdir:
                if os.path.exists('dataset/'+data+sub):
                    shutil.rmtree('dataset/'+data+sub)
                os.makedirs('dataset/'+data+sub)

    for i in range(3):
        print(f"{subdir[i*2][:-1]} : {len(os.listdir(directory+subdir[i*2]))}")
        print()

        l=[]
        for j in range(len(os.listdir(directory+subdir[i*2]))):
            l.append(plt.imread(directory+subdir[i*2]+os.listdir(directory+subdir[i*2])[j]))

        l2=[]
        for j in range(len(os.listdir(directory+subdir[i*2+1]))):
            l2.append(plt.imread(directory+subdir[i*2+1]+os.listdir(directory+subdir[i*2+1])[j]))

        X_train, X_test, y_train, y_test = train_test_split(l, l2, train_size=train_test_split_)

        X_aug=[] ; y_aug=[] ;

        for j in range(400-len(X_train)):
            if j<len(X_train):
                image,mask = heavy_augmentation(X_train[j],y_train[j])
                X_aug.append(image) ; y_aug.append(mask)
            else:
                k = j
                while(k>len(X_train)):
                    k-=len(X_train)
                image,mask = heavy_augmentation(X_train[k-len(X_train)],y_train[k-len(X_train)])
                X_aug.append(image) ; y_aug.append(mask)

        X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, train_size=train_valid_split_)
        X_train2, X_val2, y_train2, y_val2 = train_test_split(X_aug, y_aug, train_size=train_valid_split_)

        for j in range(len(X_train1)):
            plt.imsave("dataset/train/"+subdir[i*2]+str(j)+".jpeg", X_train1[j])
    #             plt.imsave("dataset/train/"+subdir[i*2+1]+str(j)+".jpeg", y_train1[j].reshape(256, 256))
            Image.fromarray(y_train1[j]).save("dataset/train/"+subdir[i*2+1]+str(j)+".jpeg")

        for j in range(len(X_train2)):
            plt.imsave("dataset/train/"+subdir[i*2]+str(len(X_train1)+j)+".jpeg", X_train2[j])
    #             plt.imsave("dataset/train/"+subdir[i*2+1]+str(len(X_train1)+j)+".jpeg", y_train2[j].reshape(256, 256))
            Image.fromarray(y_train2[j]).save("dataset/train/"+subdir[i*2+1]+str(len(X_train1)+j)+".jpeg")

        for j in range(len(X_val1)):
            plt.imsave("dataset/validation/"+subdir[i*2]+str(j)+".jpeg", X_val1[j])
    #             plt.imsave("dataset/validation/"+subdir[i*2+1]+str(j)+".jpeg", y_val1[j].reshape(256, 256))
            Image.fromarray(y_val1[j]).save("dataset/validation/"+subdir[i*2+1]+str(j)+".jpeg")

        for j in range(len(X_val2)):
            plt.imsave("dataset/validation/"+subdir[i*2]+str(j+len(X_val1))+".jpeg", X_val2[j])
    #             plt.imsave("dataset/validation/"+subdir[i*2+1]+str(j+len(X_val1))+".jpeg", y_val2[j].reshape(256, 256))
            Image.fromarray(y_val2[j]).save("dataset/validation/"+subdir[i*2+1]+str(j+len(X_val1))+".jpeg")

        for j in range(len(X_test)):
            plt.imsave("dataset/test/"+subdir[i*2]+str(j)+".jpeg", X_test[j])
    #             plt.imsave("dataset/test/"+subdir[i*2+1]+str(j)+".jpeg", y_test[j].reshape(256, 256))
            Image.fromarray(y_test[j]).save("dataset/test/"+subdir[i*2+1]+str(j)+".jpeg")


# In[11]:


StoreData(directory,subdir)


# In[12]:


print(len(os.listdir('dataset/train/benign_image')))
print(len(os.listdir('dataset/train/malignant_image')))
print(len(os.listdir('dataset/train/normal_image')))
print()
print(len(os.listdir('dataset/test/benign_image')))
print(len(os.listdir('dataset/test/malignant_image')))
print(len(os.listdir('dataset/test/normal_image')))
print()
print(len(os.listdir('dataset/validation/benign_image')))
print(len(os.listdir('dataset/validation/malignant_image')))
print(len(os.listdir('dataset/validation/normal_image')))

