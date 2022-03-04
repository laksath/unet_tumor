#!/usr/bin/env python
# coding: utf-8

# In[1]:


epochs_=10
batch_size_=15


# In[283]:


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


# In[284]:


########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################


# In[285]:


directory="before_augment/"
subdir=["benign_image/","benign_mask/","malignant_image/","malignant_mask/","normal_image/","normal_mask/"]
dataset=['dataset/','train/','test/','validation/']


# In[286]:


########################################################################################################################
####################################        train_test_valid      ######################################################
########################################################################################################################


# In[121]:


train_test_valid=[[[],[]],[[],[]],[[],[]]]

for i in range(1,len(dataset)):
    for j in range(3):
        for k in range(len(os.listdir(dataset[0]+dataset[i]+subdir[j*2]))):
            train_test_valid[i-1][0].append(plt.imread(dataset[0]+dataset[i]+subdir[j*2]+str(k)+".jpeg"))
            train_test_valid[i-1][1].append(plt.imread(dataset[0]+dataset[i]+subdir[j*2+1]+str(k)+".jpeg"))

X_train= np.asarray(train_test_valid[0][0],dtype=np.float32)/255
y_train= np.asarray(train_test_valid[0][1],dtype=np.float32)/255
X_test = np.asarray(train_test_valid[1][0],dtype=np.float32)/255
y_test = np.asarray(train_test_valid[1][1],dtype=np.float32)/255
X_valid= np.asarray(train_test_valid[2][0],dtype=np.float32)/255
y_valid= np.asarray(train_test_valid[2][1],dtype=np.float32)/255

X_train,y_train=shuffle(X_train, y_train)
X_test,y_test=shuffle(X_test, y_test)
X_valid,y_valid=shuffle(X_valid, y_valid)


# In[125]:


########################################################################################################################
####################################        SegregateData        #######################################################
########################################################################################################################


# In[126]:


def SegregateData(dataset,subdir):
    
    l=[[[[],[]],[[],[]],[[],[]]],[[[],[]],[[],[]],[[],[]]],[[[],[]],[[],[]],[[],[]]]]
    
    for i in range(1,4):
        for k in range(3):
            l1=[]
            for j in range(len(os.listdir(dataset[0]+dataset[i]+subdir[k*2]))):
                l1.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2]+os.listdir(dataset[0]+dataset[i]+subdir[k*2])[j]))

            l2=[]
            for j in range(len(os.listdir(dataset[0]+dataset[i]+subdir[k*2+1]))):
                l2.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2+1]+os.listdir(dataset[0]+dataset[i]+subdir[k*2+1])[j]))
            
            l[i-1][k][0]=l1
            l[i-1][k][1]=l2
    return l

l=SegregateData(dataset,subdir)

# X_train_benign        --> l[0][0][0]
# y_train_benign        --> l[0][0][1]
# X_train_malgiant      --> l[0][1][0]
# y_train_malgiant      --> l[0][1][1]
# X_train_normal        --> l[0][2][0]
# y_train_normal        --> l[0][2][1]

# X_test_benign         --> l[1][0][0]
# y_test_benign         --> l[1][0][1]
# X_test_malgiant       --> l[1][1][0]
# y_test_malgiant       --> l[1][1][1]
# X_test_normal         --> l[1][2][0]
# y_test_normal         --> l[1][2][1]

# X_validation_benign   --> l[2][0][0]
# y_validation_benign   --> l[2][0][1]
# X_validation_malgiant --> l[2][1][0]
# y_validation_malgiant --> l[2][1][1]
# X_validation_normal   --> l[2][2][0]
# y_validation_normal   --> l[2][2][1]


# In[127]:


########################################################################################################################
####################################         UNET MODEL          #######################################################
########################################################################################################################


# In[128]:


def conv_block(input_, num_filters):
    
#     conv2D_1 = Conv2D(filters = num_filters,kernel_size =  3, padding="same")(input_)
    conv2D_1 = Conv2D(filters = num_filters,kernel_size =  3, kernel_initializer = 'he_normal', padding="same")(input_)
    batch1 = BatchNormalization()(conv2D_1)
    act1 = Activation("relu")(batch1)

#     conv2D_2 = Conv2D(filters = num_filters,kernel_size =  3, padding="same")(act1)
    conv2D_2 = Conv2D(filters = num_filters,kernel_size =  3, kernel_initializer = 'he_normal', padding="same")(act1)
    batch2 = BatchNormalization()(conv2D_2)
    act2 = Activation("relu")(batch2)

    return act2

# count=0

def encoder_block(input_, num_filters):
#     global count
#     count+=1
#     print(count)
    conv = conv_block(input_, num_filters)
#     if count==4:
#         drop = Dropout(0.075)(conv)
#         pool = MaxPool2D((2, 2))(drop)
#         count=0
#         return conv, pool
    pool = MaxPool2D((2, 2))(conv)
    drop = Dropout(0.075)(pool)
    return conv, drop

def decoder_block(input_, skip_features, num_filters):
    x = Conv2DTranspose(filters = num_filters,kernel_size = (2, 2), strides=2, padding="same")(input_)
    x = Concatenate()([x, skip_features])
    x = Dropout(0.075)(x)
    x = conv_block(x, num_filters)
    return x

def unet_build(input_shape):
    
    inputs = Input(input_shape)
    
    conv1, pool1 = encoder_block(inputs, 16)
    conv2, pool2 = encoder_block(pool1, 32)
    conv3, pool3 = encoder_block(pool2, 64) 
    conv4, pool4 = encoder_block(pool3, 128) 

    bridge = conv_block(pool4, 256)

    decoder_1 = decoder_block(bridge, conv4, 128)
    decoder_2 = decoder_block(decoder_1, conv3, 64)
    decoder_3 = decoder_block(decoder_2, conv2, 32)
    decoder_4 = decoder_block(decoder_3, conv1, 16)

    outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid") (decoder_4)
#     outputs = Conv2D(1, 1, padding="same") (decoder_4)

    model = Model(inputs, outputs, name="U-Net")
    return model

def improvised_unet_build(input_shape):
    
    inputs = Input(input_shape)
    
    conv1, pool1 = encoder_block(inputs, 16)
    conv2, pool2 = encoder_block(pool1, 32)
    conv3, pool3 = encoder_block(pool2, 64) 
    conv4, pool4 = encoder_block(pool3, 128) 

    bridge = conv_block(pool4, 256)

    decoder_1 = decoder_block(bridge, conv4, 128)
    decoder_2 = decoder_block(decoder_1, conv3, 64)
    decoder_3 = decoder_block(decoder_2, conv2, 32)
    
#     decoder_4 = decoder_block(decoder_3, conv1, 16)

    x = Conv2DTranspose(filters = 16,kernel_size = (2, 2), strides=2, padding="same")(decoder_3)
    x = Concatenate()([x, conv1])
    x = Dropout(0.075)(x)
    
    x1 = conv_block(x, 16)
    y1_output = Conv2D(1, 1, padding="same", activation="sigmoid",name='grayscale') (x1)

    x2 = conv_block(x, 16)
    y2_output = Conv2D(3, 1, padding="same", activation="sigmoid",name='colour') (x2)
    
    model = Model(inputs=inputs, outputs=[y1_output, y2_output], name="U-Net")
    return model
#     outputs = Conv2D(1, 1, padding="same") (decoder_4)

def DiceAccuracy(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    
    intersection = keras.sum(targets*inputs,keepdims=True)
    dice = (2*intersection + smooth) / (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice


# In[129]:


input_shape = (256, 256, 3)
unet_model = unet_build(input_shape)
unet_model.summary()

# input_shape = (256, 256, 3)
# improvised_unet_model = improvised_unet_build(input_shape)
# improvised_unet_model.summary()


# In[130]:


unet_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[DiceAccuracy])

# improvised_unet_model.compile(optimizer='adam',
#                               loss={'grayscale': 'binary_crossentropy', 'colour': 'mse'},
#                               metrics={'grayscale': DiceAccuracy, 'colour': tf.keras.metrics.Accuracy()},
#                               )


# In[131]:


filepath='_BCE-1e-3-x0_new'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_DiceAccuracy',mode='max',verbose=1,save_best_only=True)
callbacks = [checkpoint]


# In[133]:


history = unet_model.fit(X_train,y_train,batch_size=batch_size_,epochs=epochs_,validation_data=(X_valid,y_valid),callbacks=callbacks)
# history = improvised_unet_model.fit(X_train,(y_train,X_train) ,batch_size=25,epochs=300,validation_data=(X_valid,(y_valid,X_valid)),callbacks=callbacks)


# In[180]:


model_best = load_model(filepath,custom_objects={'DiceAccuracy':DiceAccuracy,})


# In[251]:


def predict_data():
    
    results=['results','train','test','valid']
    
    dirs=[('/test/test_tumor/','/test/test_true/','/test/test_predicted/'),
      ('/train/train_tumor/','/train/train_true/','/train/train_predicted/'),
      ('/valid/valid_tumor/','/valid/valid_true/','/valid/valid_predicted/')
     ]
    
    data_saves=[(X_test,y_test),(X_train,y_train),(X_valid,y_valid)]
    
    for x in range(len(results)):
        if(x==0):
            if os.path.exists(results[x]):
                shutil.rmtree(results[x])
            os.makedirs(results[x])
        else:
            if os.path.exists(results[0]+"/"+results[x]):
                shutil.rmtree(results[0]+"/"+results[x])
            os.makedirs(results[0]+"/"+results[x])

    for i in range(3):
        for j in range(3):
            if os.path.exists("results"+dirs[i][j]):
                shutil.rmtree("results"+dirs[i][j])
            os.makedirs("results"+dirs[i][j])

            if(j==0):
                for k in range(len(data_saves[i][0])):
                    plt.imsave("results"+dirs[i][j]+str(k+1)+".jpeg", data_saves[i][0][k])
            if(j==1):
                for k in range(len(data_saves[i][1])):
                    Image.fromarray((data_saves[i][1][k] * 255).astype(np.uint8).reshape(256, 256)).save("results"+dirs[i][j]+str(k+1)+".jpeg")
            if(j==2):
                pred=model_best.predict(data_saves[i][0])
                for k in range(len(pred)):
                    Image.fromarray((pred[k] * 255).astype(np.uint8).reshape(256, 256)).save("results"+dirs[i][j]+str(k+1)+".jpeg")


# In[1]:


def predict_classwise():
    
    category_= ['train/','test/','validation/']
    tumor    = ['benign/','malginant/','normal/']
    dirs_    = ['tumor/','true/','predicted/']

    if os.path.exists("classwise_results/"):
            shutil.rmtree("classwise_results/")
    os.makedirs("classwise_results/")

    for i in range(3):
        if os.path.exists("classwise_results/"+category_[i]):
            shutil.rmtree("classwise_results/"+category_[i])
        os.makedirs("classwise_results/"+category_[i])
        for j in range(3):
            if os.path.exists("classwise_results/"+category_[i]+tumor[j]):
                shutil.rmtree("classwise_results/"+category_[i]+tumor[j])
            os.makedirs("classwise_results/"+category_[i]+tumor[j])
            for k in range(3):
                path_="classwise_results/"+category_[i]+tumor[j]+dirs_[k]
                if os.path.exists(path_):
                    shutil.rmtree(path_)
                os.makedirs(path_)

                if(k==0):
                    X=np.asarray(l[i][j][0],dtype=np.float32)/255
                    for m in range(len(X)):
                        plt.imsave(path_ + str(m) + ".jpeg", X[m])
                if(k==1):
                    y = np.asarray(l[i][j][1],dtype=np.float32)/255
                    for m in range(len(y)):
                        Image.fromarray((y[m]* 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")
                if(k==2):     
                    X=np.asarray(l[i][j][0],dtype=np.float32)/255
                    pred = model_best.predict(X)
                    for m in range(len(pred)):
                        Image.fromarray((pred[m]* 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")
                    print(path_+" : "+str(DiceAccuracy(np.asarray(l[i][j][1],dtype=np.float32)/255,pred).numpy()[0])+"\n")


# In[2]:


predict_data()
predict_classwise()




