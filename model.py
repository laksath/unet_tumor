import os.path
import os
import time
import shutil
from sklearn.utils import shuffle
from tensorflow.keras import backend as keras
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian
import math
from numpy.random import seed
from tensorflow.keras.utils import to_categorical
from collections import Counter
import gc

tf.compat.v1.disable_eager_execution()

epochs_ = 1000
batch_size_ = 32

filepath = '/workspace/data/code_j/m3a1e1000b32seed6_pyramid_4/'
histpath = '/workspace/data/code_j/m3a1e1000b32seed6_pyramid_4.xlsx'



def setseed(seedn):
    seed(seedn)
    tf.random.set_seed(seedn)
    tf.keras.utils.set_random_seed(seedn)
    os.environ['PYTHONHASHSEED'] = str(seedn)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

setseed(6)


# weights = [1/3,1/3,1/3] 
#weights = [1/2,1/4,1/4]
#weights = [1/4,1/2,1/4]
#weights = [1/4,1/4,1/2]
# weights = [1,1,1]

weights = [0.25,0.25,0.5]



#unet variants :  https://arxiv.org/ftp/arxiv/papers/2011/2011.01118.pdf

# In[284]:

########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################

subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
# dataset = ['/workspace/data/code_j/dataset2/',
#            'train/', 'test/', 'validation/']
dataset = ['/workspace/data/code_j/dataset/',
           'train/', 'test/', 'validation/']

# In[286]:
########################################################################################################################
####################################        train_test_valid      ######################################################
########################################################################################################################

train_test_valid = [[[], [], []], [[], [], []], [[], [], []]]

for i in range(1, len(dataset)):
    for j in range(3):
    # for j in range(2):
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
                l1.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2]+dir_l[j]))
            # l1 = [skimage.transform.resize(image, IMAGE_DIMENSION) for image in l1]
            
            l2 = []
            for j in range(len(dir_l2)):
                l2.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2+1]+dir_l2[j]))
            # new_dim = int(256/IMAGE_DIMENSION[0])
            # l2 = [image[::new_dim,::new_dim] for image in l2]

            l3=[]
            for j in range(len(dir_l2)):
                q=[0,0,0]
                q[k]=1
                l3.append(q)

            l[i-1][k][0] = np.asarray(l1, dtype=np.float32)/255
            l[i-1][k][1] = np.asarray(l2, dtype=np.float32)/255
            l[i-1][k][2] = l3

    return l

l = SegregateData(dataset, subdir)
# In[127]:

########################################################################################################################
####################################         UNET MODEL          #######################################################
########################################################################################################################

def original_improvised_unet_build():

    def conv_block(input_, num_filters):
        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
                        kernel_initializer='he_normal', padding="same")(input_)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)
        conv2D_2 = Conv2D(filters=num_filters, kernel_size=3,
                        kernel_initializer='he_normal', padding="same")(act1)
        batch2 = BatchNormalization()(conv2D_2)
        act2 = Activation("relu")(batch2)
        return act2

    def encoder_block(input_, num_filters):
        conv = conv_block(input_, num_filters)
        pool = MaxPool2D((2, 2))(conv)
        drop = Dropout(0.3)(pool)
        return conv, drop

    def decoder_block(input_, skip_features, num_filters):
        x = Conv2DTranspose(filters=num_filters, kernel_size=(2, 2), strides=2, padding="same")(input_)
        x = Concatenate()([x, skip_features])
        x = Dropout(0.3)(x)
        x = conv_block(x, num_filters)
        return x

    def b1(input_,num_filters):
        x1 = conv_block(input_, num_filters)
        y1_output = Conv2D(1, 1, padding="same", activation="sigmoid", name='grayscale')(x1)
        return y1_output

    def b2(input_,num_filters):
        x2 = conv_block(input_, num_filters)
        y2_output = Conv2D(3, 1, padding="same", activation="sigmoid", name='colour')(x2)
        return y2_output

    def b3(input_,num_filters):

        pool1 = MaxPool2D((2, 2))(input_)
        pool2 = MaxPool2D((2, 2))(pool1)
        pool3 = MaxPool2D((2, 2))(pool2)

        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
            kernel_initializer='he_normal', padding="same")(pool3)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)

        glb = GlobalAveragePooling2D(name='avg_pool')(act1)
        dense1 = Dense(1000, activation='relu')(glb)
        dense2 = Dense(3)(dense1)
        y3_output = tf.keras.layers.Activation('softmax', name='classifier')(dense2)
        return y3_output

    inputs = Input((256,256,3))

    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)

    bridge = conv_block(pool4, 1024)

    decoder_1 = decoder_block(bridge, conv4, 512)
    decoder_2 = decoder_block(decoder_1, conv3, 256)
    decoder_3 = decoder_block(decoder_2, conv2, 128)

    x = Conv2DTranspose(filters=64, kernel_size=(
        2, 2), strides=2, padding="same")(decoder_3)
    x = Concatenate()([x, conv1])
    x = Dropout(0.4)(x)


    branch1 = b1(x,64)

    # if b2 is needed , uncomment b2; 
    # if b3 is needed , uncomment b3; 
    # if both are needed , uncomment both; 

    branch2 = b2(x,64)
    # branch3 = b3(x,32)

    # model = Model(inputs=inputs, outputs=[branch1,], name="Improvised_UNet")
    model = Model(inputs=inputs, outputs=[branch1,branch2], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1,branch3], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1, branch2, branch3], name="Improvised_UNet")
    return model

def improvised_unet_build():

    def conv_block(input_, num_filters):
        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
                        kernel_initializer='he_normal', padding="same")(input_)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)
        conv2D_2 = Conv2D(filters=num_filters, kernel_size=3,
                        kernel_initializer='he_normal', padding="same")(act1)
        batch2 = BatchNormalization()(conv2D_2)
        act2 = Activation("relu")(batch2)
        return act2

    def encoder_block(input_, num_filters):
        conv = conv_block(input_, num_filters)
        pool = MaxPool2D((2, 2))(conv)
        drop = Dropout(0.3)(pool)
        return conv, drop

    def decoder_block(input_, skip_features, num_filters):
        x = Conv2DTranspose(filters=num_filters, kernel_size=(
            2, 2), strides=2, padding="same")(input_)
        x = Concatenate()([x, skip_features])
        x = Dropout(0.3)(x)
        x = conv_block(x, num_filters)
        return x

    def b1(input_,num_filters):
        x1 = conv_block(input_, num_filters)
        y1_output = Conv2D(1, 1, padding="same",
                        activation="sigmoid", name='grayscale')(x1)
        return y1_output

    def b2(input_,num_filters):
        x2 = conv_block(input_, num_filters)
        y2_output = Conv2D(3, 1, padding="same",
                        activation="sigmoid", name='colour')(x2)
        return y2_output

    def b3(input_,num_filters):

        pool1 = MaxPool2D((2, 2))(input_)
        pool2 = MaxPool2D((2, 2))(pool1)
        pool3 = MaxPool2D((2, 2))(pool2)

        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
            kernel_initializer='he_normal', padding="same")(pool3)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)

        glb = GlobalAveragePooling2D(name='avg_pool')(act1)
        dense1 = Dense(1000, activation='relu')(glb)
        dense2 = Dense(3)(dense1)
        y3_output = tf.keras.layers.Activation('softmax', name='classifier')(dense2)
        return y3_output

    # inputs = Input((256,256,3))
    inputs = Input((288,288,3))

    conv1, pool1 = encoder_block(inputs, 16)
    conv2, pool2 = encoder_block(pool1, 32)
    conv3, pool3 = encoder_block(pool2, 64)
    conv4, pool4 = encoder_block(pool3, 128)

    bridge = conv_block(pool4, 256)

    decoder_1 = decoder_block(bridge, conv4, 128)
    decoder_2 = decoder_block(decoder_1, conv3, 64)
    decoder_3 = decoder_block(decoder_2, conv2, 32)

    x = Conv2DTranspose(filters=16, kernel_size=(
        2, 2), strides=2, padding="same")(decoder_3)
    x = Concatenate()([x, conv1])
    x = Dropout(0.4)(x)


    branch1 = b1(x,16)

    # if b2 is needed , uncomment b2; 
    # if b3 is needed , uncomment b3; 
    # if both are needed , uncomment both; 

    # branch2 = b2(x,16)
    # branch3 = b3(x,32)

    model = Model(inputs=inputs, outputs=[branch1,], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1,branch2], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1,branch3], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1, branch2, branch3], name="Improvised_UNet")
    return model


def Imp_attnUNet():
    #https://www.kaggle.com/code/chaitanyasriarimanda/brainmri-image-segmentation-attention-unet/notebook
    def conv_block(inp,filters):
        x=Conv2D(filters,(3,3),padding='same',activation='relu')(inp)
        x=Conv2D(filters,(3,3),padding='same')(x)
        x=BatchNormalization(axis=3)(x)
        x=Activation('relu')(x)
        return x

    def encoder_block(inp,filters):
        x=conv_block(inp,filters)
        p=MaxPooling2D(pool_size=(2,2))(x)
        return x,p

    def attention_block(l_layer,h_layer): #Attention Block
        phi=Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer)
        theta=Conv2D(h_layer.shape[-1],(1,1),strides=(2,2),padding='same')(h_layer)
        x=add([phi,theta])
        x=Activation('relu')(x)
        x=Conv2D(1,(1,1),padding='same',activation='sigmoid')(x)
        x=UpSampling2D(size=(2,2))(x)
        x=multiply([h_layer,x])
        x=BatchNormalization(axis=3)(x)
        return x
        
    def decoder_block(inp,filters,concat_layer):
        x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp)
        concat_layer=attention_block(inp,concat_layer)
        x=concatenate([x,concat_layer])
        x=conv_block(x,filters)
        return x 

    def b1(input_,num_filters):
        x1 = conv_block(input_, num_filters)
        y1_output = Conv2D(1, 1, padding="same",
                        activation="sigmoid", name='grayscale')(x1)
        return y1_output

    def b2(input_,num_filters):
        x2 = conv_block(input_, num_filters)
        y2_output = Conv2D(3, 1, padding="same",
                        activation="sigmoid", name='colour')(x2)
        return y2_output

    def b3(input_,num_filters):

        pool1 = MaxPool2D((2, 2))(input_)
        pool2 = MaxPool2D((2, 2))(pool1)
        pool3 = MaxPool2D((2, 2))(pool2)

        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
            kernel_initializer='he_normal', padding="same")(pool3)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)

        glb = GlobalAveragePooling2D(name='avg_pool')(act1)
        dense1 = Dense(1000, activation='relu')(glb)
        dense2 = Dense(3)(dense1)
        y3_output = tf.keras.layers.Activation('softmax', name='classifier')(dense2)
        return y3_output

    inputs=Input((256,256,3))
    d1,p1=encoder_block(inputs,64)
    d2,p2=encoder_block(p1,128)
    d3,p3=encoder_block(p2,256)
    d4,p4=encoder_block(p3,512)
    bridge=conv_block(p4,1024)
    e2=decoder_block(bridge,512,d4)
    e3=decoder_block(e2,256,d3)
    e4=decoder_block(e3,128,d2)
    # e5=decoder_block(e4,64,d1)

    x=Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(e4)
    concat_layer=attention_block(e4,d1)
    x=concatenate([x,concat_layer])

    # outputs = Conv2D(1, (1,1),activation="sigmoid")(e5)
    # model=Model(inputs=[inputs], outputs=[outputs],name='AttnetionUnet')

    branch1 = b1(x,64)

    # if b2 is needed , uncomment b2; 
    # if b3 is needed , uncomment b3; 
    # if both are needed , uncomment both; 

    branch2 = b2(x,64)
    branch3 = b3(x,32)

    model = Model(inputs=inputs, outputs=[branch1, branch2, branch3], name="Improvised_Attn_UNet")

    return model


def ImpResUNet():
    #https://www.kaggle.com/code/firqaaa/residual-unet-keras-implementation/notebook
    def bn_act(x, act=True):
        x = BatchNormalization()(x)
        if act == True:
            x = Activation("relu")(x)
        return x 

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = bn_act(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv 

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = Add()([conv, shortcut])
        return output 

    def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])
        return c

    def b1_(input_,num_filters):
        x1 = conv_block(input_, num_filters)
        y1_output = Conv2D(1, 1, padding="same",
                        activation="sigmoid", name='grayscale')(x1)
        return y1_output

    def b2_(input_,num_filters):
        x2 = conv_block(input_, num_filters)
        y2_output = Conv2D(3, 1, padding="same",
                        activation="sigmoid", name='colour')(x2)
        return y2_output

    def b3_(input_,num_filters):

        pool1 = MaxPool2D((2, 2))(input_)
        pool2 = MaxPool2D((2, 2))(pool1)
        pool3 = MaxPool2D((2, 2))(pool2)

        conv2D_1 = Conv2D(filters=num_filters, kernel_size=3,
            kernel_initializer='he_normal', padding="same")(pool3)
        batch1 = BatchNormalization()(conv2D_1)
        act1 = Activation("relu")(batch1)

        glb = GlobalAveragePooling2D(name='avg_pool')(act1)
        dense1 = Dense(1000, activation='relu')(glb)
        dense2 = Dense(3)(dense1)
        y3_output = tf.keras.layers.Activation('softmax', name='classifier')(dense2)
        return y3_output

    def ResUNet():
        f = [16, 32, 64, 128, 256]
        inputs = Input((256, 256, 3))

        ## ENCODER 
        e0 = inputs
        e1 = stem(e0, f[0])
        e2 = residual_block(e1, f[1], strides=2)
        e3 = residual_block(e2, f[2], strides=2)
        e4 = residual_block(e3, f[3], strides=2)
        e5 = residual_block(e4, f[4], strides=2)

        # BRIDGE
        b0 = conv_block(e5, f[4], strides=1)
        b1 = conv_block(b0, f[4], strides=1)

        # DECODER 
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block(u1, f[4])

        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block(u2, f[3])

        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block(u3, f[2])

        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block(u4, f[1])

        # outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)

        branch1 = b1_(d4,16)

        # if b2 is needed , uncomment b2; 
        # if b3 is needed , uncomment b3; 
        # if both are needed , uncomment both; 

        branch2 = b2_(d4,16)
        branch3 = b3_(d4,32)

        # model = Model(inputs=inputs, outputs=[branch1,], name="Improvised_UNet")
        # model = Model(inputs=inputs, outputs=[branch1,branch2], name="Improvised_UNet")
        # model = Model(inputs=inputs, outputs=[branch1,branch3], name="Improvised_UNet")
        model = Model(inputs=inputs, outputs=[branch1, branch2, branch3], name="ImprovisedRes_UNet")

        return model
    
    model = ResUNet()
    return model

def ResUNet():
    #https://www.kaggle.com/code/firqaaa/residual-unet-keras-implementation/notebook
    def bn_act(x, act=True):
        x = BatchNormalization()(x)
        if act == True:
            x = Activation("relu")(x)
        return x 

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = bn_act(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv 

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = Add()([conv, shortcut])
        return output 

    def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])
        return c
    
    def ResUNet():
        f = [16, 32, 64, 128, 256]
        inputs = Input((256, 256, 3))

        ## ENCODER 
        e0 = inputs
        e1 = stem(e0, f[0])
        e2 = residual_block(e1, f[1], strides=2)
        e3 = residual_block(e2, f[2], strides=2)
        e4 = residual_block(e3, f[3], strides=2)
        e5 = residual_block(e4, f[4], strides=2)

        # BRIDGE
        b0 = conv_block(e5, f[4], strides=1)
        b1 = conv_block(b0, f[4], strides=1)

        # DECODER 
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block(u1, f[4])

        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block(u2, f[3])

        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block(u3, f[2])

        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block(u4, f[1])

        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
        model = Model(inputs, outputs)
        return model
    
    model = ResUNet()
    return model

def PSPNet():
    def conv_block(X,filters,block):
        b = 'block_'+str(block)+'_'
        f1,f2,f3 = filters
        X_skip = X
        # block_a
        X = Convolution2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
                        padding='same',kernel_initializer='he_normal',name=b+'a')(X)
        X = BatchNormalization(name=b+'batch_norm_a')(X)
        X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)
        # # block_b
        # X = Convolution2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
        #                 padding='same',kernel_initializer='he_normal',name=b+'b')(X)
        # X = BatchNormalization(name=b+'batch_norm_b')(X)
        # X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)
       # block_c
        X = Convolution2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
                        padding='same',kernel_initializer='he_normal',name=b+'c')(X)
        X = BatchNormalization(name=b+'batch_norm_c')(X)
        #skip_conv
        X_skip = Convolution2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)
        X_skip = BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)
        # block_c + skip_conv
        X = Add(name=b+'add')([X,X_skip])
        X = ReLU(name=b+'relu')(X)
        return X
    
    def base_feature_maps(input_layer):
        # base covolution module to get input image feature maps 
        # block_1
        base = conv_block(input_layer,[32,32,64],'1')
        # # block_2
        base = conv_block(base,[64,64,128],'2')
        # # block_3
        base = conv_block(base,[128,128,256],'3')
        return base

    def pyramid_feature_maps(input_layer):
        # pyramid pooling module
        base = base_feature_maps(input_layer)
        # black
        black = GlobalAveragePooling2D(name='black_pool')(base)
        black = tf.keras.layers.Reshape((1,1,256))(black)
        black = Conv2D(filters=64,kernel_size=(1,1),name='black_1_by_1')(black)
        black = UpSampling2D(size=256,interpolation='bilinear',name='black_upsampling')(black)
        # white
        white = AveragePooling2D(pool_size=(2,2),name='white_pool')(base)
        white = Conv2D(filters=64,kernel_size=(1,1),name='white_1_by_1')(white)
        white = UpSampling2D(size=2,interpolation='bilinear',name='white_upsampling')(white)
        # base + black + white
        return concatenate([base,black,white])

    def last_conv_module():
        inputs = Input((256, 256, 3))
        X = pyramid_feature_maps(inputs)
        X = Convolution2D(filters=3,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)
        X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
        X = Activation('sigmoid' ,name='last_conv_relu')(X)
        X = Conv2D(1, (1, 1), activation='sigmoid')(X)
        # X = Flatten(name='last_conv_flatten')(X)
        model = Model(inputs=inputs, outputs=X, name="PSPNet")
        return model
    
    return last_conv_module()


def InceptionUNet():
    #https://github.com/mribrahim/inception-unet/blob/master/unetV2.py
    
    def block(prevlayer, a, b, pooling):
        conva = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
        conva = BatchNormalization()(conva)
        conva = Conv2D(b, (3, 3), activation='relu', padding='same')(conva)
        conva = BatchNormalization()(conva)
        if True == pooling:
            conva = MaxPooling2D(pool_size=(2, 2))(conva)
        
        
        convb = Conv2D(a, (5, 5), activation='relu', padding='same')(prevlayer)
        convb = BatchNormalization()(convb)
        convb = Conv2D(b, (5, 5), activation='relu', padding='same')(convb)
        convb = BatchNormalization()(convb)
        if True == pooling:
            convb = MaxPooling2D(pool_size=(2, 2))(convb)

        convc = Conv2D(b, (1, 1), activation='relu', padding='same')(prevlayer)
        convc = BatchNormalization()(convc)
        if True == pooling:
            convc = MaxPooling2D(pool_size=(2, 2))(convc)
            
        convd = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
        convd = BatchNormalization()(convd)
        convd = Conv2D(b, (1, 1), activation='relu', padding='same')(convd)
        convd = BatchNormalization()(convd)
        if True == pooling:
            convd = MaxPooling2D(pool_size=(2, 2))(convd)
            
        up = concatenate([conva, convb, convc, convd])
        return up

    img_rows = 256
    img_cols = 256
    depth = 3

    def get_unet_plus_inception():
        inputs = Input((img_rows, img_cols, depth))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        xx1 = block(inputs, 16, 16, False)
        xx2 = block(xx1, 32, 32, True)
        xx3 = block(xx2, 64, 64, True)
        xx4 = block(xx3, 128, 128, True)
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4, xx4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3, xx3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2, xx2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1, xx1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)


        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        model = Model(inputs=[inputs], outputs=[conv10])
        
        return model

    return get_unet_plus_inception()

########################################################################################################################
####################################        LOSS FUNCTIONS         #####################################################
########################################################################################################################


def DiceAccuracy(targets, inputs, smooth=1):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    intersection = keras.sum(targets*inputs)
    dice = (2*intersection + smooth) / (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return (1-DiceAccuracy(y_true, y_pred))

def DiceCoeff(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.cast(keras.greater(keras.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * keras.sum(intersection) + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
    return score

def mse_loss(imageA, imageB):
    size_ = tf.size(imageA)
    return weights[1]*(keras.sum((imageA - imageB) ** 2)/ tf.cast(size_, tf.float32))


def cce_loss(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return weights[2]*cce(y_true,y_pred)/16.118095

def weighted_hausdorff_distance(y_true, y_pred):
    w=256
    h=256
    alpha=1

    def cdist(A, B):
        # squablack norms of each row in A and B
        na = tf.blackuce_sum(tf.square(A), 1)
        nb = tf.blackuce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = keras.reshape(y_true, [w, h])
            gt_points = keras.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = keras.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(keras.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.blackuce_sum(p)

            tmp = keras.min(d_matrix, 1)

            tmp = tf.where(tf.math.is_inf(tmp), tf.zeros_like(tmp), tmp)
            term_1 = (1 / (num_est_pts + eps)) * keras.sum(p * tmp)
            d_div_p = keras.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)

            d_div_p = keras.clip(d_div_p, 0, max_dist)

            term_2 = keras.mean(d_div_p, axis=0)

            x=tf.cast(tf.math.count_nonzero(y_pred)/(255*255),tf.float32)
            term_2 = tf.where(tf.math.is_nan(term_2), x, term_2)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return keras.mean(tf.stack(batched_losses))

    return hausdorff_loss(y_true, y_pred)

# def lr_scheduler(epoch, lr):
#     return 4*0.001 * math.pow(0.75, math.floor(epoch/200))

setseed(i)

# def py():
#     def conv_block(X,filters,block):
#         # resiudal block with dilated convolutions
#         # add skip connection at last after doing convoluion operation to input X
        
#         b = 'block_'+str(block)+'_'
#         f1,f2,f3 = filters
#         X_skip = X
#         # block_a
#         X = Convolution2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
#                         padding='same',kernel_initializer='he_normal',name=b+'a')(X)
#         X = BatchNormalization(name=b+'batch_norm_a')(X)
#         X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)
#         # block_b
#         X = Convolution2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
#                         padding='same',kernel_initializer='he_normal',name=b+'b')(X)
#         X = BatchNormalization(name=b+'batch_norm_b')(X)
#         X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)
#         # block_c
#         X = Convolution2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
#                         padding='same',kernel_initializer='he_normal',name=b+'c')(X)
#         X = BatchNormalization(name=b+'batch_norm_c')(X)
#         # skip_conv
#         X_skip = Convolution2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)
#         X_skip = BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)
#         # block_c + skip_conv
#         X = Add(name=b+'add')([X,X_skip])
#         X = ReLU(name=b+'relu')(X)
#         return X
    
#     def base_feature_maps(input_layer):
#         # base covolution module to get input image feature maps 
        
#         # block_1
#         base = conv_block(input_layer,[32,32,64],'1')
#         # block_2
#         base = conv_block(base,[64,64,128],'2')
#         # block_3
#         base = conv_block(base,[128,128,256],'3')
#         return base

#     def pyramid_feature_maps(input_layer):
#         # pyramid pooling module
        
#         base = base_feature_maps(input_layer)
#         # red
#         red = GlobalAveragePooling2D(name='red_pool')(base)
#         red = tf.keras.layers.Reshape((1,1,256))(red)
#         red = Convolution2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(red)
#         red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(red)
#         # yellow
#         yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
#         yellow = Convolution2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(yellow)
#         yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)
#         # blue
#         blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
#         blue = Convolution2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)
#         blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)
#         # green
#         green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
#         green = Convolution2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(green)
#         green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)
#         # base + red + yellow + blue + green
#         return tf.keras.layers.concatenate([base,red,yellow,blue,green])

#     def last_conv_module(input_layer):
#         X = pyramid_feature_maps(input_layer)
#         X = Convolution2D(filters=1,kernel_size=1,padding='same',name='last_conv_3_by_3')(X)
#         X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
#         X = Activation('sigmoid',name='last_conv_relu')(X)
#         return X
    
#     input_layer = Input((256, 256, 3))
#     output_layer = last_conv_module(input_layer)
#     model = tf.keras.Model(inputs=input_layer,outputs=output_layer)
#     return model

from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, Add,
                                     Activation, Input, ZeroPadding2D, Flatten, GlobalAveragePooling2D, Dropout,
                                     UpSampling2D, Concatenate, Reshape, Softmax, ReLU)
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def ResNet50(input_shape=None, input_tensor=None, num_classes=None, dr=(1,1)):

    def ConvBlock(x, filters, kernel=(1,1), strides=(1,1), dr=(1,1), stage=None, block=None, ident=True):
        '''
            dr: tuple, dilitation_rate, e.g. (1,1)
            ident: bool, if ident == true => downsampling
        '''
        # name
        name = 'conv' + str(stage) + '_block' + str(block)

        # filters
        f1, f2, f3 = filters

        # save inital input value
        x_initial = x

        x = Conv2D(filters=f1, kernel_size=(1,1), strides=strides, name=name+'_1_conv')(x)
        x = BatchNormalization(name=name + '_1_bn')(x)
        x = ReLU(name=name + '_1_relu')(x)

        x = Conv2D(filters=f2, kernel_size=kernel, strides=(1,1), padding='same', dilation_rate=dr, name=name+'_2_conv')(x)
        x = BatchNormalization(name=name + '_2_bn')(x)
        x = ReLU(name=name + '_2_relu')(x)

        x = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), name=name+'_3_conv')(x)
        x = BatchNormalization(name=name + '_3_bn')(x)

        # reshape input dim to output dim
        if ident == False:
            x_initial = Conv2D(filters=f3, kernel_size=(1,1), strides=strides, name=name+'_0_conv')(x_initial)
            x_initial = BatchNormalization(name=name + '_0_bn')(x_initial)

        # add the input value x to the output and apply ReLU function
        x = Add(name=name + '_add')([x_initial, x])
        x = ReLU(name=name + '_out')(x)

        return x

    
    '''
        dr: tuple, dilitation_rate, e.g. (1,1)
    '''

    if input_tensor is None:
        input = Input(shape = input_shape, name = 'input')
    elif tf.keras.backend.is_keras_tensor(input_tensor):
        input = Input(shape = input_tensor.shape[1:], name = 'input')

    # INPUT
    x = ZeroPadding2D(padding = (3,3), name = 'conv1_pad')(input)

    # STAGE 1
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='valid', name='conv1_conv')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(name='conv1_relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2, 2), name='pool1_pool')(x)

    # STAGE 2
    x = ConvBlock(x, filters=[64, 64, 256], strides=(1,1), dr=dr, stage=2, block=1, ident=False)
    x = ConvBlock(x, filters=[64, 64, 256], dr=dr, stage=2, block=2)
    x = ConvBlock(x, filters=[64, 64, 256], dr=dr, stage=2, block=3)

    # STAGE 3
    x = ConvBlock(x, filters=[128,128,512], strides=(2,2), dr=dr, stage=3, block=1, ident=False)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=2)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=3)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=4)

    # STAGE 4
    x = ConvBlock(x, filters=[256, 256, 1024], strides=(2,2), dr = dr, stage=4, block=1, ident=False)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=2)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=3)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=4)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=5)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=6)

    # STAGE 5
    x = ConvBlock(x, filters=[512, 512, 2048], strides=(2,2), dr=dr, stage=5, block=1, ident = False)
    x = ConvBlock(x, filters=[512, 512, 2048], dr=dr, stage=5, block=2)
    x = ConvBlock(x, filters=[512, 512, 2048], dr=dr, stage=5, block=3)

    # OUTPUT
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    output = Dense(units = num_classes,  activation='softmax', name='predictions')(x)

    model = Model(inputs = [input], outputs = [output], name = 'ResNet50')

    return model

def PSPNet(input_shape, num_classes, pool_sizes):
    
    def PyramidPooling(input_shape = None, input_tensor = None, pool_sizes = None):
        '''pool_sizes = [(1,1), (2,2), (3,3), (6,6)]'''

        if input_tensor is None:
            input = Input(shape = input_shape)
        elif tf.keras.backend.is_keras_tensor(input_tensor):
            input = Input(shape = input_tensor.shape[1:])

        _, input_height, input_width, input_depth = input.shape

        N = len(pool_sizes) # number pyramid levels

        f = input_depth // N # pyramid level depth
        
        output = [input]

        for level, pool_size in enumerate(pool_sizes):
            x = AveragePooling2D(pool_size = pool_size, padding = 'same', name='avg_'+str(level))(input)
            x = Conv2D(filters = f, kernel_size = (1,1), name='conv_'+str(level))(x)
            x = UpSampling2D(size = pool_size, interpolation = 'bilinear', name='bilinear_'+str(level))(x)

            # Crop/pad to input size
            # Pooling with padding = 'same': After upsamling => size > input size => crop
            # Pooling with padding = 'valid': After upsamling => size < input size => pad
            height, width = x.shape[1:3]
            dif_height, dif_width = height - input_height, width - input_width
            x = tf.keras.layers.Cropping2D(cropping = ((0,dif_height), (0,dif_width)), name='crop_'+str(level))(x)

            output += [x]

        # concatenate input and pyramid levels
        output = Concatenate(name='concatenate')(output)

        model = Model(inputs = [input], outputs = [output], name = 'pyramid_module')

        return model
    
    def Decoder(input_shape = None, input_tensor = None, target_shape = None, num_classes = None):
        '''target_shape: (H,W)'''

        if input_tensor is None:
            input = Input(shape = input_shape)
        elif tf.keras.backend.is_keras_tensor(input_tensor):
            input = Input(shape = input_tensor.shape[1:])

        height, width = target_shape

        x = Conv2D(filters = 512, kernel_size = (3,3))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dropout(.1)(x)

        # x = Conv2D(filters = num_classes, kernel_size = (1,1))(x)
        x = Conv2D(filters = num_classes, kernel_size = (1,1), activation='sigmoid')(x)
        x = Resizing(height = height, width = width, interpolation = 'bilinear', name = 'resize')(x)
        
        # output = Softmax(name = 'segmentation')(x)
        # model = Model(inputs = [input], outputs = [output], name = 'decoder')

        model = Model(inputs = [input], outputs = [x], name = 'decoder')

        return model

    input = Input(shape = input_shape, name = 'input')

    target_shape = input_shape[:2] # (H,W,B) => (H,W)

    # truncate resnet after 3. stage
    resnet = ResNet50(input_tensor=input, num_classes=num_classes, dr=(2,2))
    resnet_output = resnet.get_layer('conv3_block4_out')
    basemodel = Model(inputs = resnet.input, outputs = resnet_output.output, name = 'resnet')

    pyramid_module = PyramidPooling(input_tensor=basemodel.output, pool_sizes=pool_sizes)

    decoder = Decoder(input_tensor = pyramid_module.output, target_shape = target_shape, num_classes = num_classes)

    x = basemodel(input)
    x = pyramid_module(x)
    output = decoder(x)

    pspnet = Model(inputs = [input], outputs = [output], name = 'pspnet')

    return pspnet



# from segmentation_models import PSPNet
# import segmentation_models as sm
# sm.set_framework('tf.keras')

# PSPNet_model = py()
PSPNet_model = PSPNet((256, 256, 3), 1, [(1,1), (2,2), (3,3), (6,6)])
# PSPNet_model = sm.PSPNet(input_shape=(288, 288, 3), classes=1, activation='sigmoid')
PSPNet_model.summary()

class TestCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

        def on_epoch_end(self, epoch, logs=None):
            
            # X_valid3, y_valid3 = shuffle(X_valid, y_valid, random_state=5)
            # pred_valid = PSPNet_model.predict(X_valid3)
            
            # pred_valid_32 = np.array_split(pred_valid, len(pred_valid)/32)
            # y_valid3_32 = np.array_split(y_valid3, len(y_valid3)/32)
            
            # logs['val_dice'] = sum(DiceAccuracy(pred_valid_32[x], y_valid3_32[x]) for x in range(len(pred_valid_32)))/len(pred_valid_32)
            
            pred_valid = PSPNet_model.predict(X_valid)
            logs['val_dice'] = DiceAccuracy(pred_valid[:32], y_valid[:32])

callbacks = [ModelCheckpoint(filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
            # TestCallback()
            ]

PSPNet_model.compile(
    optimizer='adam',
    loss=dice_loss,
    metrics=[DiceAccuracy],
)

# sm.utils.set_trainable(PSPNet_model, recompile=True)

# PSPNet_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss=dice_loss,
#     metrics=[DiceAccuracy],
# )


history = PSPNet_model.fit(X_train, y_train, batch_size=batch_size_,
                            epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)
hist = pd.DataFrame(history.history)
hist.to_excel(histpath)

# add weights at dl

# for i in [6]:
#     setseed(i)
    
#     if os.path.exists('/workspace/data/op2/T_m3_original_a1e1000b32b13seed'+str(i)):
#         shutil.rmtree('/workspace/data/op2/T_m3_original_a1e1000b32b13seed'+str(i))
#     os.makedirs('/workspace/data/op2/T_m3_original_a1e1000b32b13seed'+str(i))

#     original_improvised_unet_model = original_improvised_unet_build()
#     filepath = '/workspace/data/op2/T_m3_original_a1e1000b32b13seed'+str(i)+'/m3_original_a1e1000b32b12_final_seed'+str(i)+'/'
#     histpath = '/workspace/data/op2/T_m3_original_a1e1000b32b13seed'+str(i)+'/m3_original_a1e1000b32b12_final_seed'+str(i)+'.xlsx'

#     original_improvised_unet_model.compile(
#                             optimizer='adam',
#                             loss={
#                                     'grayscale': dice_loss,
#                                     'colour': mse_loss,
#                                     # 'classifier': cce_loss,
#                                     # 'contour': weighted_hausdorff_distance,
#                                 },
#                             metrics={
#                                         'grayscale': DiceAccuracy,
#                                         'colour': ['accuracy'],
#                                         # 'classifier': ['accuracy'],
#                                         # 'contour': DiceAccuracy,
#                                 },
#                             )
    
#     class TestCallback(tf.keras.callbacks.Callback):
#         def __init__(self, l):
#             super().__init__()

#         # X_train_benign   --> l[0][0][0]  # X_test_benign   --> l[1][0][0]  # X_validation_benign   --> l[2][0][0]
#         # y_train_benign   --> l[0][0][1]  # y_test_benign   --> l[1][0][1]  # y_validation_benign   --> l[2][0][1]
#         # X_train_malgiant --> l[0][1][0]  # X_test_malgiant --> l[1][1][0]  # X_validation_malgiant --> l[2][1][0]
#         # y_train_malgiant --> l[0][1][1]  # y_test_malgiant --> l[1][1][1]  # y_validation_malgiant --> l[2][1][1]
#         # X_train_normal   --> l[0][2][0]  # X_test_normal   --> l[1][2][0]  # X_validation_normal   --> l[2][2][0]
#         # y_train_normal   --> l[0][2][1]  # y_test_normal   --> l[1][2][1]  # y_validation_normal   --> l[2][2][1]
        
#         def on_epoch_end(self, epoch, logs=None):
            
#             pred_test_benign = original_improvised_unet_model.predict(l[1][0][0])
#             pred_test_malignant = original_improvised_unet_model.predict(l[1][1][0])
#             pred_test_normal = original_improvised_unet_model.predict(l[1][2][0])
            
#             pred_validation_benign = original_improvised_unet_model.predict(l[2][0][0])
#             pred_validation_malignant = original_improvised_unet_model.predict(l[2][1][0])
#             pred_validation_normal = original_improvised_unet_model.predict(l[2][2][0])
            
#             y_valid_pred = original_improvised_unet_model.predict(X_valid)
#             y_test_pred = original_improvised_unet_model.predict(X_test)
            
#             def DiceCoeff_npy(y_true, y_pred, smooth=1e-6):
#                 y_true_f = y_true.flatten()
#                 y_pred_f = (y_pred.flatten()>0.5).astype(np.float32)
#                 intersection = y_true_f * y_pred_f
#                 score = (2. * np.sum(intersection) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
#                 return score
            
#             logs['val_dice_ben'] = sum([DiceCoeff_npy(l[2][0][1][i],pred_validation_benign[0][i]) for i in range(len(l[2][0][1]))])/len(l[2][0][1])
#             logs['val_dice_mal'] = sum([DiceCoeff_npy(l[2][1][1][i],pred_validation_malignant[0][i]) for i in range(len(l[2][1][1]))])/len(l[2][1][1])
#             logs['val_dice_nor'] = sum([DiceCoeff_npy(l[2][2][1][i],pred_validation_normal[0][i]) for i in range(len(l[2][2][1]))])/len(l[2][2][1])
            
#             logs['test_dice_ben'] = sum([DiceCoeff_npy(l[1][0][1][i],pred_test_benign[0][i]) for i in range(len(l[1][0][1]))])/len(l[1][0][1])
#             logs['test_dice_mal'] = sum([DiceCoeff_npy(l[1][1][1][i],pred_test_malignant[0][i]) for i in range(len(l[1][1][1]))])/len(l[1][1][1])
#             logs['test_dice_nor'] = sum([DiceCoeff_npy(l[1][2][1][i],pred_test_normal[0][i]) for i in range(len(l[1][2][1]))])/len(l[1][2][1])
            
#             logs['valid_dice'] = DiceCoeff(y_valid,y_valid_pred[0]).numpy()
#             logs['test_dice'] = DiceCoeff(y_test,y_test_pred[0]).numpy()
                                
#             logs['valid_dice_npy'] = sum([DiceCoeff_npy(y_valid[i],y_valid_pred[0][i]) for i in range(len(y_valid))])/len(y_valid)
#             logs['test_dice_npy'] =  sum([DiceCoeff_npy(y_test[i],y_test_pred[0][i]) for i in range(len(y_test))])/len(y_test)
                                
#             logs['test_reconstruction'] = mse_loss(X_test,y_test_pred[1])
            
#     callbacks = [
#         ModelCheckpoint(filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
#         TestCallback(l=l),
#         ReduceLROnPlateau(monitor='grayscale_loss', factor=0.2, patience=150 , min_lr=1e-7),
#     ]
    
#     history = original_improvised_unet_model.fit(X_train, (y_train, X_train), batch_size=batch_size_,
#                                         epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid)), callbacks=callbacks)

#     del original_improvised_unet_model
#     keras.clear_session()
#     gc.collect()

#     hist = pd.DataFrame(history.history)
    # hist.to_excel(histpath)

# for i in [9]:
#     setseed(i)
#     folder_name = '/workspace/data/op2/m3a1e1000b32seed'+str(i)+'_InceptionUNet'
#     if os.path.exists(folder_name):
#         shutil.rmtree(folder_name)
#     os.makedirs(folder_name)

#     InceptionUNet_model = InceptionUNet()

#     filepath = folder_name+'/m3a1e1000b32seed'+str(i)+'_InceptionUNet/'
#     histpath = folder_name+'/m3a1e1000b32seed'+str(i)+'_InceptionUNet.xlsx'

#     callbacks = [ModelCheckpoint(
#         filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
#     InceptionUNet_model.compile(
#         optimizer='adam',
#         loss=dice_loss,
#         metrics=[DiceAccuracy]
#     )
#     history = InceptionUNet_model.fit(X_train, y_train, batch_size=batch_size_,
#                             epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

#     del InceptionUNet_model
#     keras.clear_session()
#     gc.collect()

#     hist = pd.DataFrame(history.history)
#     hist.to_excel(histpath)


# ResUNet_model = ResUNet()
# callbacks = [ModelCheckpoint(
#   filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
# ResUNet_model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=[DiceAccuracy]
# )
# history = ResUNet_model.fit(X_train, y_train, batch_size=batch_size_,
#                             epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

# AUNet = attnUNet()
# callbacks = [ModelCheckpoint(
#   filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
# AUNet.compile(
#     optimizer='adam',
#     loss=dice_loss,
#     metrics=[DiceAccuracy]
# )
# history = AUNet.fit(X_train, y_train, batch_size=batch_size_,
#                         epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)


# b1 only :

# for i in range(2):
#     setseed(5)

#     if(i==0):
#         filepath = '/workspace/data/code_j/trial1/'
#         histpath = '/workspace/data/code_j/trial1.xlsx'
#     if(i==1):
#         filepath = '/workspace/data/code_j/trial2/'
#         histpath = '/workspace/data/code_j/trial2.xlsx'

#     improvised_unet_model = improvised_unet_build()

#     improvised_unet_model.compile(
#                             optimizer='adam',
#                             loss={'grayscale': dice_loss,
#                                 # 'colour': mse_loss,
#                                 # 'classifier': 'categorical_crossentropy',
#                                 # 'contour': weighted_hausdorff_distance,
#                                 },
#                             metrics={'grayscale': DiceAccuracy,
#                                     # 'colour': ['accuracy'],
#                                     # 'classifier': ['accuracy'],
#                                     # 'contour': DiceAccuracy,
#                                     },
#                             )
                                
#     callbacks = [ModelCheckpoint(
#     filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

#     history = improvised_unet_model.fit(X_train, (y_train, ), batch_size=batch_size_,
#                                         epochs=epochs_, validation_data=(X_valid, (y_valid, )), callbacks=callbacks)

#     del improvised_unet_model
#     keras.clear_session()
#     gc.collect()

#     hist = pd.DataFrame(history.history)
#     hist.to_excel(histpath)

# b1 and b2 :

# improvised_unet_model = improvised_unet_build()
# improvised_unet_model.summary()

# improvised_unet_model.compile(
#                         optimizer='adam',
#                         loss={'grayscale': dice_loss,
#                               'colour': mse_loss,
#                             # 'classifier': 'categorical_crossentropy',
#                             # 'contour': weighted_hausdorff_distance,
#                             },
#                         metrics={'grayscale': DiceAccuracy,
#                                 'colour': ['accuracy'],
#                                 # 'classifier': ['accuracy'],
#                                 # 'contour': DiceAccuracy,
#                                 },
#                         )
                            
# callbacks = [ModelCheckpoint(
# filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

# history = improvised_unet_model.fit(X_train, (y_train, X_train), batch_size=batch_size_,
#                                     epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid)), callbacks=callbacks)

# b1 and b3 :

# improvised_unet_model = improvised_unet_build()
# improvised_unet_model.summary()

# improvised_unet_model.compile(
#                         optimizer='adam',
#                         loss={'grayscale': dice_loss,
#                             #   'colour': mse_loss,
#                             'classifier': 'categorical_crossentropy',
#                             # 'contour': weighted_hausdorff_distance,
#                             },
#                         metrics={'grayscale': DiceAccuracy,
#                                 # 'colour': ['accuracy'],
#                                 'classifier': ['accuracy'],
#                                 # 'contour': DiceAccuracy,
#                                 },
#                         )
                            
# callbacks = [ModelCheckpoint(
# filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

# history = improvised_unet_model.fit(X_train, (y_train , y_train2), batch_size=batch_size_,
#                                     epochs=epochs_, validation_data=(X_valid, (y_valid, y_valid2)), callbacks=callbacks)


# b1 , b2 and b3  :

# for i in [6]:
#     setseed(i)
    
#     if os.path.exists('/workspace/data/op2/m3_original_a1e1000b32b12seed'+str(i)):
#         shutil.rmtree('/workspace/data/op2/m3_original_a1e1000b32b12seed'+str(i))
#     os.makedirs('/workspace/data/op2/m3_original_a1e1000b32b12seed'+str(i))

#     original_improvised_unet_model = original_improvised_unet_build()
#     filepath = '/workspace/data/op2/m3_original_a1e1000b32b12seed'+str(i)+'/m3_original_a1e1000b32b12seed'+str(i)+'/'
#     histpath = '/workspace/data/op2/m3_original_a1e1000b32b12seed'+str(i)+'/m3_original_a1e1000b32b12seed'+str(i)+'.xlsx'

#     original_improvised_unet_model.compile(
#                             optimizer='adam',
#                             loss={
#                                     'grayscale': dice_loss,
#                                     'colour': mse_loss,
#                                     # 'classifier': cce_loss,
#                                     # 'contour': weighted_hausdorff_distance,
#                                 },
#                             metrics={
#                                         'grayscale': DiceAccuracy,
#                                         'colour': ['accuracy'],
#                                         # 'classifier': ['accuracy'],
#                                         # 'contour': DiceAccuracy,
#                                 },
#                             )
                                
#     callbacks = [
#         ModelCheckpoint(filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
#         # tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
#     ]
#     history = original_improvised_unet_model.fit(X_train, (y_train, X_train, ), batch_size=batch_size_,
#                                         epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, )), callbacks=callbacks)

#     history = original_improvised_unet_model.fit(X_train, (y_train, X_train, ), batch_size=batch_size_,
#                                         epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, )), callbacks=callbacks)

#     del original_improvised_unet_model
#     keras.clear_session()
#     gc.collect()

#     hist = pd.DataFrame(history.history)
#     hist.to_excel(histpath)


# for q in range(2):
#     for i in [2,3,4,5,6,7,8,9]:
#         weights = [[1/3,1/3,1/3],[1/2,1/4,1/4]][q]
#         weights_str = ["33_33_33","50_25_25"][q]
#         setseed(i)
        
#         if os.path.exists('/workspace/data/op2/m3_original_a1e1000b32seed'+str(i)+weights_str):
#             shutil.rmtree('/workspace/data/op2/m3_original_a1e1000b32seed'+str(i)+weights_str)
#         os.makedirs('/workspace/data/op2/m3_original_a1e1000b32seed'+str(i)+weights_str)

#         original_improvised_unet_model = original_improvised_unet_build()
#         filepath = '/workspace/data/op2/m3_original_a1e1000b32seed'+str(i)+weights_str+'/m3_original_a1e1000b32seed'+str(i)+weights_str+'/'
#         histpath = '/workspace/data/op2/m3_original_a1e1000b32seed'+str(i)+weights_str+'/m3_original_a1e1000b32seed'+str(i)+weights_str+'.xlsx'

#         original_improvised_unet_model.compile(
#                                 optimizer='adam',
#                                 loss={
#                                         'grayscale': dice_loss,
#                                         'colour': mse_loss,
#                                         'classifier': cce_loss,
#                                         # 'contour': weighted_hausdorff_distance,
#                                     },
#                                 metrics={
#                                             'grayscale': DiceAccuracy,
#                                             'colour': ['accuracy'],
#                                             'classifier': ['accuracy'],
#                                             # 'contour': DiceAccuracy,
#                                     },
#                                 )
                                    
#         callbacks = [
#             ModelCheckpoint(filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
#             # tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
#         ]
#         history = original_improvised_unet_model.fit(X_train, (y_train, X_train, y_train2), batch_size=batch_size_,
#                                             epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, y_valid2)), callbacks=callbacks)

#         del original_improvised_unet_model
#         keras.clear_session()
#         gc.collect()

#         hist = pd.DataFrame(history.history)
#         hist.to_excel(histpath)

# for q in range(2):
#     for i in [9]:
#         weights = [[1/3,1/3,1/3],[1/2,1/4,1/4]][q]
#         weights_str = ["33_33_33","50_25_25"][q]
#         setseed(i)
        
#         folder_name = '/workspace/data/op2/m3a1e1000b32seed'+str(i)+'_Attn_'+weights_str
#         if os.path.exists(folder_name):
#             shutil.rmtree(folder_name)
#         os.makedirs(folder_name)

#         improvised_Attnunet_model = Imp_attnUNet()
#         filepath = folder_name+'/m3a1e1000b32seed'+str(i)+'_'+weights_str+'/'
#         histpath = folder_name+'/m3a1e1000b32seed'+str(i)+'_'+weights_str+'.xlsx'

#         improvised_Attnunet_model.compile(
#                                 optimizer='adam',
#                                 loss={'grayscale': dice_loss,
#                                     'colour': mse_loss,
#                                     'classifier': cce_loss,
#                                     # 'contour': weighted_hausdorff_distance,
#                                     },
#                                 metrics={'grayscale': DiceAccuracy,
#                                         'colour': ['accuracy'],
#                                         'classifier': ['accuracy'],
#                                         # 'contour': DiceAccuracy,
#                                         },
#                                 )
                                    
#         callbacks = [
#             ModelCheckpoint(filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
#             # tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
#         ]
#         history = improvised_Attnunet_model.fit(X_train, (y_train, X_train, y_train2), batch_size=batch_size_,
#                                             epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, y_valid2)), callbacks=callbacks)

#         del improvised_Attnunet_model
#         keras.clear_session()
#         gc.collect()

#         hist = pd.DataFrame(history.history)
#         hist.to_excel(histpath)




print("TRAIN COMPLETE")