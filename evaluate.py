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
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import classification_report

# In[2]:


# weights = [1/3,1/3,1/3]
# weights = [1/2,1/4,1/4]
# weights = [1/4,1/2,1/4]
weights = [1/4,1/4,1/2]

model_number = 1
x = os.getcwd()
filepath = x+"/"+x.split("/")[-1]+"/"
path_x = x+"/classwise_results/"
print(filepath)

# In[3]:

subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/op/m3a1e1000b32x_lr_50_25_25_result/dataset/',
           'train/', 'test/', 'validation/']


# In[4]:

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
print(str(Counter(y_train2.tolist()))+" "+str(Counter(y_test2.tolist()))+" "+str(Counter(y_valid2.tolist())))

y_valid2 = to_categorical(y_valid2)
y_test2 = to_categorical(y_test2)
y_train2 = to_categorical(y_train2)


# In[5]:

########################################################################################################################
####################################        SegregateData        #######################################################
########################################################################################################################

def SegregateData(dataset, subdir):

    l = [
            [
                [[], [], []],
                [[], [], []],
                [[], [], []],
            ], 
            [
                [[], [], []],
                [[], [], []],
                [[], [], []],
            ],
            [
                [[], [], []],
                [[], [], []],
                [[], [], []],
            ],
        ]

    for i in range(1, 4):
        for k in range(3):
            dir_l = os.listdir(dataset[0]+dataset[i]+subdir[k*2])
            dir_l2 = os.listdir(dataset[0]+dataset[i]+subdir[k*2+1])

            l1 = []
            for j in range(len(dir_l)):
                l1.append(plt.imread(
                    dataset[0]+dataset[i]+subdir[k*2]+dir_l[j]))

            l2 = []
            for j in range(len(dir_l2)):
                l2.append(plt.imread(
                    dataset[0]+dataset[i]+subdir[k*2+1]+dir_l2[j]))

            l3=[]
            for j in range(len(dir_l2)):
                q=[0,0,0]
                q[k]=1
                l3.append(q)

            l[i-1][k][0] = l1
            l[i-1][k][1] = l2
            l[i-1][k][2] = l3

    return l


l = SegregateData(dataset, subdir)

# X_train_benign   --> l[0][0][0]  # X_test_benign   --> l[1][0][0]  # X_validation_benign   --> l[2][0][0]
# y_train_benign   --> l[0][0][1]  # y_test_benign   --> l[1][0][1]  # y_validation_benign   --> l[2][0][1]
# X_train_malgiant --> l[0][1][0]  # X_test_malgiant --> l[1][1][0]  # X_validation_malgiant --> l[2][1][0]
# y_train_malgiant --> l[0][1][1]  # y_test_malgiant --> l[1][1][1]  # y_validation_malgiant --> l[2][1][1]
# X_train_normal   --> l[0][2][0]  # X_test_normal   --> l[1][2][0]  # X_validation_normal   --> l[2][2][0]
# y_train_normal   --> l[0][2][1]  # y_test_normal   --> l[1][2][1]  # y_validation_normal   --> l[2][2][1]


# In[6]:


def DiceAccuracy(targets, inputs, smooth=1e-1):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    intersection = keras.sum(targets*inputs)
    dice = (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def DiceCoeff(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.cast(keras.greater(keras.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * keras.sum(intersection) + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pblack):
    return 1-DiceAccuracy(y_true, y_pblack)

def mse_loss(imageA, imageB):
    size_ = tf.size(imageA)
    return weights[1]*(keras.sum((imageA - imageB) ** 2)/ tf.cast(size_, tf.float32))


def cce_loss(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return weights[2]*cce(y_true,y_pred)/16.118095

def mse_eval(imageA, imageB):
    size_ = tf.size(imageA)
    return keras.sum((imageA - imageB) ** 2)/ tf.cast(size_, tf.float32)

model_best = load_model(filepath, custom_objects={
                        'DiceAccuracy': DiceAccuracy,
                        'dice_loss' : dice_loss,
                        'mse_loss': mse_loss,
                        'cce_loss': cce_loss,
                        })

# In[7]:
def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def branch1_op(i,j,pred,path2,path_name,f):

    f.write(path_name)
    print(path_name)

    for m in range(len(pred[0])):
        Image.fromarray((pred[0][m] * 255).astype(np.uint8).reshape(256, 256)).save(path2 + str(m) + ".jpeg")
    
    pred_b1 = []
    for m in range(len(os.listdir(path2))):
        pred_b1.append(plt.imread(path2+str(m)+".jpeg"))
    pred_b1 = np.asarray(pred_b1, dtype=np.float32)/255

    b1=0
    for m in range(len(l[i][j][1])):
        true_val = np.asarray(l[i][j][1][m], dtype=np.float32)/255
        b1+=DiceCoeff(true_val,pred_b1[m]).numpy()
    b1/=len(l[i][j][1])
    f.write('branch1 : '+str(b1)+'\n')
    print('branch1 : '+str(b1))


def branch2_op(X,pred,path3,f):

    for m in range(len(pred[1])):
        plt.imsave(path3 + str(m) + ".jpeg", pred[1][m])

    b2 = mse_eval(X,pred[1]).numpy()
    f.write('branch2 : '+str(b2)+' (mse loss)\n')
    print('branch2 : '+str(b2)+' (mse loss)')


def branch3_op(i,j,pred,path_name,s,f):

    acc = tf.keras.metrics.Accuracy()
    b3_y_true = [np.argmax(val) for val in l[i][j][2]]
    b3_y_pred = [np.argmax(val) for val in pred[2]]
    b3 = acc(b3_y_true,b3_y_pred).numpy()

    f.write('branch3 : '+str(b3)+'\n')
    print('branch3 : '+str(b3))

    s+=f'{path_name} : \n'
    for m in range(len(pred[0])):
        s+=str(pred[2][m])+' '+str(l[i][j][2][m])+'\n'
    s+='\n'
    return s

def auc_classification_branch1(true_path_,pred_path_,f):
    true_b1 = []
    for m in range(len(os.listdir(true_path_))):
        true_b1.append(plt.imread(true_path_+str(m)+".jpeg"))
    true_b1 = np.asarray(true_b1, dtype=np.float32)/255

    pred_b1 = []
    for m in range(len(os.listdir(pred_path_))):
        pred_b1.append(plt.imread(pred_path_+str(m)+".jpeg"))
    pred_b1 = np.asarray(pred_b1, dtype=np.float32)/255

    y_true = true_b1.ravel()
    y_pred = pred_b1.ravel()

    fpr, tpr, thresholds = metrics.roc_curve((y_true>0.5).astype(np.uint8), y_pred)
    roc_auc = auc(fpr,tpr)

    classification_rep  = classification_report((y_true>0.5).astype(np.uint8), (y_pred>0.5).astype(np.uint8))

    f.write("roc_auc branch1 : "+str(roc_auc)+'\n')
    f.write("classification branch1 : \n")
    f.write(str(classification_rep))

    print("roc_auc branch1 : "+str(roc_auc))
    print("classification branch1 : ")
    print(str(classification_rep))


def auc_classification_branch3(classifier_true,classifier_pred,path_title,f):
    print()
    print()
    print(path_title)
    target_names = ['benign', 'malginant', 'normal']
    for m in range(3):
        fpr, tpr, thresholds = metrics.roc_curve(classifier_true[:,m], classifier_pred[:,m])
        roc_auc = auc(fpr,tpr)

        f.write(target_names[m]+' : \n')
        f.write("roc_auc branch3 : "+str(roc_auc)+'\n')

        print(target_names[m]+' : ')
        print("roc_auc branch3 : "+str(roc_auc))

    b3_y_true = [np.argmax(val) for val in classifier_true]
    b3_y_pred = [np.argmax(val) for val in classifier_pred]

    classification_rep = classification_report(b3_y_true, b3_y_pred, target_names=target_names)
    f.write("classification branch3 : \n")
    f.write(str(classification_rep))
    print("classification branch3 : ")
    print(str(classification_rep))


def predict_classwise_auroc_precision_recall_fscore_support():
    s=''
    category_ = ['train/', 'test/', 'validation/']
    tumor = ['benign/', 'malginant/', 'normal/']
    dirs_ = ['tumor/', 'true/', 'predicted/']

    make_dir(path_x)

    if(os.path.exists(path_x+"test_results.txt")):
        open(path_x+"test_results.txt", 'w').close()
    else:
        f = open(path_x+"test_results.txt", "a")

    for i in range(3):
        make_dir(path_x+category_[i])
        classifier_pred=[] ; classifier_true=[]
        for j in range(3):
            make_dir(path_x+category_[i]+tumor[j])
            for k in range(3):
                make_dir(path_x+category_[i]+tumor[j]+dirs_[k])
            path2 = path_x+category_[i]+tumor[j]+dirs_[2]+'branch1/'
            path3 = path_x+category_[i]+tumor[j]+dirs_[2]+'branch2/'
            make_dir(path2)
            make_dir(path3)

            X = np.asarray(l[i][j][0], dtype=np.float32)/255
            y = np.asarray(l[i][j][1], dtype=np.float32)/255
            pred = model_best.predict(X)
            path_ = path_x+category_[i]+tumor[j]+dirs_[0]
            for m in range(len(X)):
                plt.imsave(path_ + str(m) + ".jpeg", X[m])
            path_ = path_x+category_[i]+tumor[j]+dirs_[1]
            for m in range(len(y)):
                Image.fromarray((y[m] * 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")

            
            for m in range(len(pred[2])):
                classifier_pred.append(pred[2][m].tolist())
                classifier_true.append(list(l[i][j][2][m]))
            
            path_name = f'{category_[i]} {tumor[j]} {dirs_[2]} : '

            branch1_op(i,j,pred,path2,path_name,f)
            if(j!=2):
                true_path_ = path_x+category_[i]+tumor[j]+'true/'
                pred_path_ = path_x+category_[i]+tumor[j]+'predicted/branch1/'
                auc_classification_branch1(true_path_,pred_path_,f)

            branch2_op(X,pred,path3,f)

            b3=branch3_op(i,j,pred,path_name,s,f)
            s+=b3
        
        classifier_pred=np.asarray(classifier_pred); classifier_true=np.asarray(classifier_true)
        auc_classification_branch3(classifier_true,classifier_pred,path_x+category_[i]+tumor[j],f)

    f.write(s+'\n')
    f.close()


predict_classwise_auroc_precision_recall_fscore_support()