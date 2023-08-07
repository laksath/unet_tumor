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
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian
import math

epochs_ = 1000
batch_size_ = 32
filepath = '/workspace/data/code_j/m3a1e1000b32x_50_25_25_3/'
histpath = '/workspace/data/code_j/m3a1e1000b32x_50_25_25_3.xlsx'

# weights = [1/3,1/3,1/3]
weights = [1/2,1/4,1/4]
# weights = [1/4,1/2,1/4]
# weights = [1/4,1/4,1/2]

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


    branch1 = b1(x,16)

    # if b2 is needed , uncomment b2; 
    # if b3 is needed , uncomment b3; 
    # if both are needed , uncomment both; 

    branch2 = b2(x,16)
    branch3 = b3(x,32)

    # model = Model(inputs=inputs, outputs=[branch1,], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1,branch2], name="Improvised_UNet")
    # model = Model(inputs=inputs, outputs=[branch1,branch3], name="Improvised_UNet")
    model = Model(inputs=inputs, outputs=[branch1, branch2, branch3], name="Improvised_UNet")
    return model


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
    dice = (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return weights[0]*(1-DiceAccuracy(y_true, y_pred))

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

# PSPNet_model = PSPNet()
# PSPNet_model.summary()
# callbacks = [ModelCheckpoint(
#     filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
# PSPNet_model.compile(
#     optimizer='adam',
#     loss=dice_loss,
#     metrics=[DiceAccuracy]
# )
# history = PSPNet_model.fit(X_train, y_train, batch_size=batch_size_,
#                             epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

# InceptionUNet_model = InceptionUNet()
# callbacks = [ModelCheckpoint(
#     filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
# InceptionUNet_model.compile(
#     optimizer='adam',
#     loss=dice_loss,
#     metrics=[DiceAccuracy]
# )
# history = InceptionUNet_model.fit(X_train, y_train, batch_size=batch_size_,
#                             epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)

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

# improvised_unet_model = improvised_unet_build()
# improvised_unet_model.summary()

# improvised_unet_model.compile(
#                         optimizer='adam',
#                         loss={'grayscale': dice_loss,
#                             # 'colour': mse_loss,
#                             # 'classifier': 'categorical_crossentropy',
#                             # 'contour': weighted_hausdorff_distance,
#                             },
#                         metrics={'grayscale': DiceAccuracy,
#                                 # 'colour': ['accuracy'],
#                                 # 'classifier': ['accuracy'],
#                                 # 'contour': DiceAccuracy,
#                                 },
#                         )
                            
# callbacks = [ModelCheckpoint(
# filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]

# history = improvised_unet_model.fit(X_train, (y_train, ), batch_size=batch_size_,
#                                     epochs=epochs_, validation_data=(X_valid, (y_valid, )), callbacks=callbacks)

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

improvised_unet_model = improvised_unet_build()
improvised_unet_model.summary()

improvised_unet_model.compile(
                        optimizer='adam',
                        loss={'grayscale': dice_loss,
                              'colour': mse_loss,
                            'classifier': cce_loss,
                            # 'contour': weighted_hausdorff_distance,
                            },
                        metrics={'grayscale': DiceAccuracy,
                                'colour': ['accuracy'],
                                'classifier': ['accuracy'],
                                # 'contour': DiceAccuracy,
                                },
                        )
                            
callbacks = [
    ModelCheckpoint(filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True),
    # tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
]

history = improvised_unet_model.fit(X_train, (y_train, X_train, y_train2), batch_size=batch_size_,
                                    epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid, y_valid2)), callbacks=callbacks)





print("END OF TRAINING")

hist = pd.DataFrame(history.history)
hist.to_excel(histpath)

print("TRAIN COMPLETE")