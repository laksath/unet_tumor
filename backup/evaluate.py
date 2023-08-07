#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import os
from sklearn.utils import shuffle
from tensorflow.keras import backend as keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import cv2

# In[3]:


model_number=1
filepath = '/workspace/data/op/m1a1e600b32_result4/m1a1e600b32_res4/'
print(filepath)


# In[4]:


subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/op/m1a1e600b32_result4/dataset/',
           'train/', 'test/', 'validation/']


# In[286]:
########################################################################################################################
####################################        train_test_valid      ######################################################
########################################################################################################################


from tensorflow.keras.utils import to_categorical

train_test_valid = [[[], [], []], [[], [], []], [[], [], []]]

for i in range(1, len(dataset)):
    for j in range(3):
        for k in range(len(os.listdir(dataset[0]+dataset[i]+subdir[j*2]))):
            train_test_valid[i-1][0].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2]+str(k)+".jpeg"))
            train_test_valid[i-1][1].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2+1]+str(k)+".jpeg"))
            train_test_valid[i-1][2].append(j)
            
X_train = np.asarray(train_test_valid[0][0], dtype=np.float32)/255
y_train = np.asarray(train_test_valid[0][1], dtype=np.float32)/255
y_train2 = np.asarray(train_test_valid[0][2])

X_test = np.asarray(train_test_valid[1][0], dtype=np.float32)/255
y_test = np.asarray(train_test_valid[1][1], dtype=np.float32)/255
y_test2 = np.asarray(train_test_valid[1][2])

X_valid = np.asarray(train_test_valid[2][0], dtype=np.float32)/255
y_valid = np.asarray(train_test_valid[2][1], dtype=np.float32)/255
y_valid2 = np.asarray(train_test_valid[2][2])

X_train, y_train, y_train2 = shuffle(X_train, y_train, y_train2, random_state=5)
X_test, y_test, y_test2 = shuffle(X_test, y_test, y_test2, random_state=5)
X_valid, y_valid, y_valid2 = shuffle(X_valid, y_valid, y_valid2, random_state=5)

from collections import Counter
print(Counter(y_train2.tolist()))
print(Counter(y_test2.tolist()))
print(Counter(y_valid2.tolist()))

y_valid2 = to_categorical(y_valid2)
y_test2 = to_categorical(y_test2)
y_train2 = to_categorical(y_train2)

def create_contours(y):
    contours_l=[]
    y3=[]
    for i in range(len(y)):
        ret, thresh = cv2.threshold(y[i], 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh.astype('uint8'), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        l = np.zeros([256,256,3])
        l[:,:,0] = np.ones([256,256])*0
        l[:,:,1] = np.ones([256,256])*0
        l[:,:,2] = np.ones([256,256])*0
        cv2.drawContours(image=l, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        l = cv2.cvtColor(np.float32(l), cv2.COLOR_BGR2GRAY)
        contours_l.append(contours)
        y3.append(l.astype(np.uint8))
        
    return [contours_l,y3]

y_train_contour=(np.array(create_contours((y_train*255).astype(np.uint8))[1])/255).astype(np.float32)
y_test_contour =(np.array(create_contours((y_test*255).astype(np.uint8))[1])/255).astype(np.float32)
y_valid_contour=(np.array(create_contours((y_valid*255).astype(np.uint8))[1])/255).astype(np.float32)

# In[125]:

########################################################################################################################
####################################        SegregateData        #######################################################
########################################################################################################################

def SegregateData(dataset, subdir):

    l = [
            [
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
            ], 
            [
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
            ],
            [
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
            ],
        ]

    for i in range(1, 4):
        for k in range(3):
            dir_l = os.listdir(dataset[0]+dataset[i]+subdir[k*2])
            dir_l2 = os.listdir(dataset[0]+dataset[i]+subdir[k*2+1])

            l1 = []
            for j in range(len(dir_l)):
                l1.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2]+dir_l[j]))

            l2 = []
            for j in range(len(dir_l2)):
                l2.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2+1]+dir_l2[j]))

            l3=[]
            for j in range(len(dir_l2)):
                q=[0,0,0]
                q[k]=1
                l3.append(q)

            l4=[]
            for j in range(len(dir_l2)):
                im1 = plt.imread(dataset[0]+dataset[i]+subdir[k*2+1]+dir_l2[j])
                ret, thresh = cv2.threshold(im1, 150, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(image=thresh.astype('uint8'), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                temp = np.zeros([256,256,3])
                temp[:,:,0] = np.ones([256,256])*0
                temp[:,:,1] = np.ones([256,256])*0
                temp[:,:,2] = np.ones([256,256])*0
                cv2.drawContours(image=l, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                temp = cv2.cvtColor(np.float32(l), cv2.COLOR_BGR2GRAY)
                temp = temp.astype(np.uint8)
                l4.append(temp)
            l[i-1][k][0] = l1
            l[i-1][k][1] = l2
            l[i-1][k][2] = l3
            l[i-1][k][3] = l4

    return l


l = SegregateData(dataset, subdir)

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

# In[6]:

def IoU_coeff(targets, inputs, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)

    intersection = keras.sum(targets*inputs)
    total = keras.sum(targets) + keras.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def IoU_Loss(y_true, y_pred):
    return IoU_coeff(y_true, y_pred)


def DiceAccuracy(targets, inputs, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    intersection = keras.sum(targets*inputs)
    dice = (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1-DiceAccuracy(y_true, y_pred)


def FocalLoss(targets, inputs, alpha=0.8, gamma=2):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    BCE = keras.binary_crossentropy(targets, inputs)
    BCE_EXP = keras.exp(-BCE)
    focal_loss = keras.mean(alpha * keras.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss

def focal_loss(y_true, y_pred):
    return FocalLoss(y_true, y_pred)


def DiceBCELoss(targets, inputs, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    intersection = keras.sum(targets*inputs)
    dice_loss = 1 - (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE

def dicebce_loss(y_true, y_pred):
    return DiceBCELoss(y_true, y_pred)


def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    TP = keras.sum((inputs * targets))
    FP = keras.sum(((1-targets) * inputs))
    FN = keras.sum((targets * (1-inputs)))
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    return 1 - Tversky

def tversky_loss(y_true, y_pred):
    return TverskyLoss(y_true, y_pred)


scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def scce_loss(y_true, y_pred):
    return scce(y_true, y_pred)

def mse_loss(imageA, imageB):
    err = keras.sum((imageA - imageB) ** 2)/(256 * 256)
    return err

model_best = load_model(filepath, custom_objects={
                        'DiceAccuracy': DiceAccuracy,
                        'dice_loss' : dice_loss,

                        'mse_loss': mse_loss,

                        'IoU_coeff' : IoU_coeff,
                        'IoU_Loss' : IoU_Loss,

                        'FocalLoss' : FocalLoss,
                        'focal_loss' : focal_loss,

                        'TverskyLoss' : TverskyLoss,
                        'tversky_loss' : tversky_loss,
                        })

# In[7]:


def predict_classwise():
    s=''
    category_ = ['train/', 'test/', 'validation/']
    tumor = ['benign/', 'malginant/', 'normal/']
    dirs_ = ['tumor/', 'true/', 'predicted/']

    path_x = "/workspace/data/op/m1a1e600b32_result4/classwise_results/"
    if os.path.exists(path_x):
        shutil.rmtree(path_x)
    os.makedirs(path_x)

    if(os.path.exists(path_x+"test_results.txt")):
        open(path_x+"test_results.txt", 'w').close()
    else:
        f = open(path_x+"test_results.txt", "a")

    for i in range(3):
        if os.path.exists(path_x+category_[i]):
            shutil.rmtree(path_x+category_[i])
        os.makedirs(path_x+category_[i])

        for j in range(3):
            if os.path.exists(path_x+category_[i]+tumor[j]):
                shutil.rmtree(path_x+category_[i]+tumor[j])
            os.makedirs(path_x+category_[i]+tumor[j])

            for k in range(3):
                path_ = path_x+category_[i]+tumor[j]+dirs_[k]
                if os.path.exists(path_):
                    shutil.rmtree(path_)
                os.makedirs(path_)
                
                if(model_number == 1 and k==2):
                    path2 = path_x+category_[i]+tumor[j]+dirs_[k]+'branch1/'
                    if os.path.exists(path2):
                        shutil.rmtree(path2)
                    os.makedirs(path2)

                    path3 = path_x+category_[i]+tumor[j]+dirs_[k]+'branch2/'
                    if os.path.exists(path3):
                        shutil.rmtree(path3)
                    os.makedirs(path3)

                if(k == 0):
                    X = np.asarray(l[i][j][0], dtype=np.float32)/255
                    for m in range(len(X)):
                        plt.imsave(path_ + str(m) + ".jpeg", X[m])
                if(k == 1):
                    y = np.asarray(l[i][j][1], dtype=np.float32)/255
                    for m in range(len(y)):
                        Image.fromarray(
                            (y[m] * 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")
                if(k == 2):

                    X = np.asarray(l[i][j][0], dtype=np.float32)/255
                    pred = []
                    if(model_number == 1):
                        pred = model_best.predict(X)
                        for m in range(len(pred[0])):
                            Image.fromarray(
                                (pred[0][m] * 255).astype(np.uint8).reshape(256, 256)).save(path2 + str(m) + ".jpeg")
                        for m in range(len(pred[1])):
                            plt.imsave(path3 + str(m) + ".jpeg", pred[1][m])

                        f.write(f'{category_[i]} {tumor[j]} {dirs_[k]} : ')
                        f.write(str(DiceAccuracy(np.asarray(l[i][j][1], dtype=np.float32)/255, pred[0]).numpy())+'\n')

                        acc = tf.keras.metrics.Accuracy()
                        f.write('branch3 : '+str(acc([np.argmax(val) for val in pred[2]],[np.argmax(val) for val in l[i][j][2]]).numpy())+'\n')
                        f.write('branch4 : '+str(DiceAccuracy(np.asarray(l[i][j][2], dtype=np.float32)/255, pred[3]).numpy())+'\n')

                        s+=f'{category_[i]} {tumor[j]} {dirs_[k]} : \n'
                        for m in range(len(pred[0])):
                            s+=str(pred[2][m])+' '+str(l[i][j][2][m])+'\n'
                        s+='\n'

                        print(path_+" : "+str(DiceAccuracy(np.asarray(l[i][j][1], dtype=np.float32)/255, pred[0]).numpy()))
                        print('branch3 : '+str(acc([np.argmax(val) for val in pred[2]],[np.argmax(val) for val in l[i][j][2]]).numpy()))

                    else:
                        pred = model_best.predict(X)
                        for m in range(len(pred)):
                            Image.fromarray(
                                (pred[m] * 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")

                        f.write(f'{category_[i]} {tumor[j]} {dirs_[k]} : ')
                        f.write(str(DiceAccuracy(np.asarray(l[i][j][1], dtype=np.float32)/255, pred).numpy())+'\n')
                        
                        print(path_+" : ")
                        print(DiceAccuracy(np.asarray(l[i][j][1], dtype=np.float32)/255, pred).numpy())

    f.write(s)
    f.write('\n')
    f.close()

predict_classwise()

print("!!!!!!!!")