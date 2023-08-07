from keras import backend as K
from keras.layers import concatenate, Conv2DTranspose, Activation
from keras.layers import BatchNormalization


from keras.layers import Conv2D, Input, AvgPool2D
from keras.models import Model

import warnings

import theano.tensor as T
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D,  MaxPooling2D, ZeroPadding2D, Dropout
from keras.layers import convolutional



import os.path
import os
import time
import shutil
from sklearn.utils import shuffle
from tensorflow.keras import backend as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian
import math

epochs_ =1000
batch_size_ = 32
model_number = 0
filepath = '/workspace/data/code_j/m1a1e1000b32_inception3/'
hist_name = 'trainHistoryaabove4'
# In[283]:

#unet variants :  https://arxiv.org/ftp/arxiv/papers/2011/2011.01118.pdf

# In[284]:


########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################


# In[285]:


subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/code_j/dataset/',
           'train/', 'test/', 'validation/']


# In[286]:
########################################################################################################################
####################################        train_test_valid      ######################################################
########################################################################################################################


# In[121]:


from tensorflow.keras.utils import to_categorical

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

# In[127]:


########################################################################################################################
####################################         UNET MODEL          #######################################################
########################################################################################################################


# In[128]:

def UNET():
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

    def unet_build():
        inputs = Input((256,256,3))

        conv1, pool1 = encoder_block(inputs, 16)
        conv2, pool2 = encoder_block(pool1, 32)
        conv3, pool3 = encoder_block(pool2, 64)
        conv4, pool4 = encoder_block(pool3, 128)

        bridge = conv_block(pool4, 256)

        decoder_1 = decoder_block(bridge, conv4, 128)
        decoder_2 = decoder_block(decoder_1, conv3, 64)
        decoder_3 = decoder_block(decoder_2, conv2, 32)
        decoder_4 = decoder_block(decoder_3, conv1, 16)

        outputs = Conv2D(1, (1, 1), padding="same",
                        activation="sigmoid")(decoder_4)

        model = Model(inputs, outputs, name="U-Net")
        return model

    def improvised_unet_build():

        inputs = Input((256,256,3))

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

        x1 = conv_block(x, 16)
        y1_output = Conv2D(1, 1, padding="same",
                        activation="sigmoid", name='grayscale')(x1)

        x2 = conv_block(x, 16)
        y2_output = Conv2D(3, 1, padding="same",
                        activation="sigmoid", name='colour')(x2)


        x3 = conv_block(x, 16)

        # MAXPOOLING HERE (256,256,32 ) -> (128,128,32)
        # MAXPOOLING HERE (128,128,32) -> (64,64,32)
        # MAXPOOLING HERE (64,64,32) -> (32,32,32)
        # CONV LAYER (32,32,32)
        # (GLOBAL AVG POOLING) (1,1,32)
        # FCN

        y3_gray = Conv2D(1, 1, padding="same",
                        activation="sigmoid")(x3)
        a = Flatten()(y3_gray)
        b = Dense(1500, activation='relu')(a)
        y3_output = Dense(3, name='classifier')(b)
        # y3_output = Dense(2, name='classifier')(b)

        x4 = conv_block(x, 16)
        y4_output = Conv2D(1, 1, padding="same",
                        activation="sigmoid", name='contour')(x4)

        # model = Model(inputs=inputs, outputs=[y1_output, y2_output], name="U-Net")
        # model = Model(inputs=inputs, outputs=[y1_output, y2_output, y3_output], name="Improvised_UNet")
        model = Model(inputs=inputs, outputs=[y1_output, y2_output, y3_output, y4_output], name="Improvised_UNet")
        return model

    if(model_number==0):
        return unet_build()

    return improvised_unet_build()


def attnUNet():
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

  inputs=Input((256,256,3))
  d1,p1=encoder_block(inputs,64)
  d2,p2=encoder_block(p1,128)
  d3,p3=encoder_block(p2,256)
  d4,p4=encoder_block(p3,512)
  b1=conv_block(p4,1024)
  e2=decoder_block(b1,512,d4)
  e3=decoder_block(e2,256,d3)
  e4=decoder_block(e3,128,d2)
  e5=decoder_block(e4,64,d1)
  outputs = Conv2D(1, (1,1),activation="sigmoid")(e5)
  model=Model(inputs=[inputs], outputs=[outputs],name='AttnetionUnet')

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

def Deeplabv2():
    #https://github.com/DavideA/deeplabv2-keras/blob/master/deeplabV2.py
    img_input = Input(shape=[256,256,3])
    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(img_input)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = tf.keras.layers.Conv2D(filters=512,kernel_size= (3 ,3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = tf.keras.layers.Conv2D(filters=512,kernel_size= ( 3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = tf.keras.layers.Conv2D(filters=1024,kernel_size= ( 3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 =tf.keras.layers.Conv2D(filters=1024,kernel_size= ( 3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = tf.keras.layers.Conv2D(filters=1024,kernel_size= ( 3, 3),dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

    s = concatenate([b1, b2, b3, b4])
    logits=UpSampling2D(size=256,interpolation='bilinear',name='white_upsampling')(s)
    output=Convolution2D(1, 1, 1, activation='softmax')(logits)
    model = Model(img_input, output, name='deeplabV2')
    return model




def Deeplabv3plus():
    #https://keras.io/examples/vision/deeplabv3_plus/
    def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,):
        x = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(block_input)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.nn.relu(x)


    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = tf.keras.layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size=1)
        return output

    def DeeplabV3Plus(image_size, num_classes):
        model_input = tf.keras.Input(shape=(image_size, image_size, 3))
        resnet50 = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)

        input_a = tf.keras.layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = tf.keras.layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
        return tf.keras.Model(inputs=model_input, outputs=model_output)


    model = DeeplabV3Plus(image_size=256, num_classes=1)
    return model

def UnetPlusPlus():
    #https://github.com/AlphaJia/keras_unet_plus_plus/blob/master/unetpp.py
    dropout_rate = 0.5

    def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):

        x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same')(input_tensor)
        x = BatchNormalization(axis=2)(x)
        x = Activation('relu')(x)

        return x


    def model_build_func(input_shape, n_labels, using_deep_supervision=False):

        nb_filter = [32,64,128,256,512]

        # Set image data format to channels first
        global bn_axis

        K.set_image_data_format("channels_last")
        bn_axis = -1
        inputs = Input(shape=input_shape, name='input_image')

        conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
        pool1 = AvgPool2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

        conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
        pool2 = AvgPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

        up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0])

        conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
        pool3 = AvgPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

        up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

        up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
        conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

        conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
        pool4 = AvgPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

        up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

        up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
        conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

        up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
        conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

        conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

        up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

        up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
        conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

        up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
        conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

        up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
        conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

        nestnet_output_1 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_1',padding='same')(conv1_2)
        nestnet_output_2 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_2', padding='same' )(conv1_3)
        nestnet_output_3 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_3', padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_4', padding='same')(conv1_5)

        if using_deep_supervision:
            model = Model(input=inputs, output=[nestnet_output_1,
                                                nestnet_output_2,
                                                nestnet_output_3,
                                                nestnet_output_4])
        else:
            model = Model(inputs=inputs, outputs=nestnet_output_4)

        return model

    return model_build_func([256,256,3],1)


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


def DiceAccuracy(targets, inputs, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    intersection = keras.sum(targets*inputs)
    dice = (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def dice_loss(y_true, y_pblack):
    return 1-DiceAccuracy(y_true, y_pblack)

def mse_loss(imageA, imageB):
    return keras.sum((imageA - imageB) ** 2)/(256 * 256)

def weighted_hausdorff_distance(y_true, y_pblack):
    w= 256
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

    def hausdorff_loss(y_true, y_pblack):
        def loss(y_true, y_pblack):
            eps = 1e-6
            y_true = keras.reshape(y_true, [w, h])
            gt_points = keras.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pblack = keras.flatten(y_pblack)
            p = y_pblack
            p_replicated = tf.squeeze(keras.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.blackuce_sum(p)

            tmp = keras.min(d_matrix, 1)

            tmp = tf.where(tf.math.is_inf(tmp), tf.zeros_like(tmp), tmp)
            term_1 = (1 / (num_est_pts + eps)) * keras.sum(p * tmp)
            d_div_p = keras.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)

            d_div_p = keras.clip(d_div_p, 0, max_dist)

            term_2 = keras.mean(d_div_p, axis=0)

            x=tf.cast(tf.math.count_nonzero(y_pblack)/(255*255),tf.float32)
            term_2 = tf.where(tf.math.is_nan(term_2), x, term_2)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pblack),
                                   dtype=tf.float32)
        return keras.mean(tf.stack(batched_losses))

    return hausdorff_loss(y_true, y_pblack)


if(model_number==0):

    #InceptionUNet_model = InceptionUNet()
    #InceptionUNet_model = UnetPlusPlus()
    InceptionUNet_model = InceptionUNet()
    print(InceptionUNet_model.summary())
    callbacks = [ModelCheckpoint(
      filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
    InceptionUNet_model.compile(
        optimizer='adam',
        loss=dice_loss,
        metrics=[DiceAccuracy]
    )
    history = InceptionUNet_model.fit(X_train, y_train, batch_size=batch_size_,
                                epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

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

    # input_shape = (256, 256, 3)
    # unet_model = unet_build(input_shape)
    # unet_model.summary()

    # unet_model.compile(
    # optimizer='adam', loss=dice_loss, metrics=[DiceAccuracy])

    # callbacks = [ModelCheckpoint(
    # filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

    # history = unet_model.fit(X_train, y_train, batch_size=batch_size_,
    #                         epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

else:
    improvised_unet_model = UNET()
    improvised_unet_model.summary()

    # improvised_unet_model.compile(
    #                             optimizer='adam',
    #                             loss={'grayscale': dice_loss,
    #                                 'colour': mse_loss,
    #                                 },
    #                             metrics={'grayscale': DiceAccuracy,
    #                                     'colour': ['accuracy'],
    #                                     },
    #                             )

    # improvised_unet_model.compile(
    #                         optimizer='adam',
    #                         loss={'grayscale': dice_loss,
    #                             'colour': mse_loss,
    #                             'classifier': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #                             'contour': dice_loss,
    #                             },
    #                         metrics={'grayscale': DiceAccuracy,
    #                                 'colour': ['accuracy'],
    #                                 'classifier': ['accuracy'],
    #                                 'contour': DiceAccuracy,
    #                                 },
    #                         )

    improvised_unet_model.compile(
                            optimizer='adam',
                            loss={'grayscale': dice_loss,
                                'colour': mse_loss,
                                'classifier': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                'contour': weighted_hausdorff_distance,
                                },
                            metrics={'grayscale': DiceAccuracy,
                                    'colour': ['accuracy'],
                                    'classifier': ['accuracy'],
                                    'contour': DiceAccuracy,
                                    },
                            )

    callbacks = [ModelCheckpoint(
    filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

    # history = improvised_unet_model.fit(X_train, (y_train, X_train), batch_size=batch_size_,
    #                                     epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid)), callbacks=callbacks)

    # history = improvised_unet_model.fit(X_train, (y_train, X_train, y_train2), batch_size=batch_size_,
    #                                     epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, y_valid2)), callbacks=callbacks)

    history = improvised_unet_model.fit(X_train, (y_train, X_train, y_train2,y_train_contour), batch_size=batch_size_,
                                        epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, y_valid2, y_valid_contour)), callbacks=callbacks)


print("END OF TRAINING")

hist = pd.DataFrame(history.history)
hist.to_excel('/workspace/data/code_j/'+hist_name+'.xlsx')

print("TRAIN COMPLETE")

time.sleep(36000)