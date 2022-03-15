#!/usr/bin/env python
# coding: utf-8

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
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

epochs_ = 301
batch_size_ = 25
model_number = 0  # 0/1/2

# In[283]:


# In[284]:


########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################


# In[285]:


directory = "/workspace/data/final_code2/before_augment/"
subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/final_code2/dataset/',
           'train/', 'test/', 'validation/']


# In[286]:
########################################################################################################################
####################################        train_test_valid      ######################################################
########################################################################################################################


# In[121]:


train_test_valid = [[[], []], [[], []], [[], []]]

for i in range(1, len(dataset)):
    for j in range(3):
        for k in range(len(os.listdir(dataset[0]+dataset[i]+subdir[j*2]))):
            train_test_valid[i-1][0].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2]+str(k)+".jpeg"))
            train_test_valid[i-1][1].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2+1]+str(k)+".jpeg"))

X_train = np.asarray(train_test_valid[0][0], dtype=np.float32)/255
y_train = np.asarray(train_test_valid[0][1], dtype=np.float32)/255
X_test = np.asarray(train_test_valid[1][0], dtype=np.float32)/255
y_test = np.asarray(train_test_valid[1][1], dtype=np.float32)/255
X_valid = np.asarray(train_test_valid[2][0], dtype=np.float32)/255
y_valid = np.asarray(train_test_valid[2][1], dtype=np.float32)/255

X_train, y_train = shuffle(X_train, y_train, random_state=5)
X_test, y_test = shuffle(X_test, y_test, random_state=5)
X_valid, y_valid = shuffle(X_valid, y_valid, random_state=5)


# In[125]:


########################################################################################################################
####################################        SegregateData        #######################################################
########################################################################################################################


# In[126]:


def SegregateData(dataset, subdir):

    l = [[[[], []], [[], []], [[], []]], [[[], []], [
        [], []], [[], []]], [[[], []], [[], []], [[], []]]]

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

            l[i-1][k][0] = l1
            l[i-1][k][1] = l2
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


# In[127]:


########################################################################################################################
####################################         UNET MODEL          #######################################################
########################################################################################################################


# In[128]:


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
    drop = Dropout(0.075)(pool)
    return conv, drop


def AttnBlock2D(x, g, inter_channel, data_format='channels_first'):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(tf.math.add(theta_x, phi_g))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = tf.math.multiply(x, rate)
    return att_x


def decoder_block(input_, skip_features, num_filters):
    x = Conv2DTranspose(filters=num_filters, kernel_size=(
        2, 2), strides=2, padding="same")(input_)
    if(model_number == 2):
        x = AttnBlock2D(x, skip_features, 1)
    else:
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

    outputs = Conv2D(1, (1, 1), padding="same",
                     activation="sigmoid")(decoder_4)

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

    x = Conv2DTranspose(filters=16, kernel_size=(
        2, 2), strides=2, padding="same")(decoder_3)
    x = Concatenate()([x, conv1])
    x = Dropout(0.075)(x)

    x1 = conv_block(x, 16)
    y1_output = Conv2D(1, 1, padding="same",
                       activation="sigmoid", name='grayscale')(x1)

    x2 = conv_block(x, 16)
    y2_output = Conv2D(3, 1, padding="same",
                       activation="sigmoid", name='colour')(x2)

    model = Model(inputs=inputs, outputs=[y1_output, y2_output], name="U-Net")
    return model

########################################################################################################################
####################################        LOSS FUNCTIONS         #####################################################
########################################################################################################################

def DiceAccuracy(targets, inputs, smooth=1e-6):
    inputs = keras.flatten(inputs)
    targets = keras.flatten(targets)
    intersection = keras.sum(targets*inputs, keepdims=True)
    dice = (2*intersection + smooth) / \
        (keras.sum(targets) + keras.sum(inputs) + smooth)
    return dice

def mse_img(imageA, imageB):
    err = keras.sum((imageA - imageB) ** 2)/(256 * 256)
    return err/10

def mse_loss():
    def mse(imageA, imageB):
        return mse_img(imageA, imageB)
    return mse


model_mse = mse_loss()

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

# In[129]:


input_shape = (256, 256, 3)
unet_model = unet_build(input_shape)
unet_model.summary()

input_shape = (256, 256, 3)
improvised_unet_model = improvised_unet_build(input_shape)
improvised_unet_model.summary()


# In[130]:


unet_model.compile(
    optimizer='adam', loss=dice_loss, metrics=[DiceAccuracy])

improvised_unet_model.compile(optimizer='adam',
                              loss={'grayscale': 'binary_crossentropy',
                                    'colour': model_mse},
                              metrics={'grayscale': DiceAccuracy,
                                       'colour': tf.keras.metrics.Accuracy()},
                              )


# In[131]:


filepath = '/workspace/trial_q'
callbacks = []
if(model_number == 0):
    callbacks = [ModelCheckpoint(
        filepath=filepath, monitor='val_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]
else:
    callbacks = [ModelCheckpoint(
        filepath=filepath, monitor='val_grayscale_DiceAccuracy', mode='max', verbose=1, save_best_only=True)]


# In[133]:

history = []

if(model_number == 0):
    history = unet_model.fit(X_train, y_train, batch_size=batch_size_,
                             epochs=epochs_, validation_data=(X_valid, y_valid), callbacks=callbacks)
else:
    history = improvised_unet_model.fit(X_train, (y_train, X_train), batch_size=batch_size_,
                                        epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid)), callbacks=callbacks)

hist = pd.DataFrame(history.history)
print(hist)
hist.to_excel('trainHistorya.xlsx')

model_best = load_model(filepath, custom_objects={
                        'DiceAccuracy': DiceAccuracy,
                        'dice_loss' : dice_loss,

                        'mse_img': mse_img,
                        'model_mse': model_mse,

                        'IoU_coeff' : IoU_coeff,
                        'IoU_Loss' : IoU_Loss,

                        'FocalLoss' : FocalLoss,
                        'focal_loss' : focal_loss,

                        'TverskyLoss' : TverskyLoss,
                        'tversky_loss' : tversky_loss,
                        })

# In[251]:


def predict_data():

    results = ['/workspace/results', 'train', 'test', 'valid']

    dirs = [('/test/test_tumor/', '/test/test_true/', '/test/test_predicted/'),
            ('/train/train_tumor/', '/train/train_true/', '/train/train_predicted/'),
            ('/valid/valid_tumor/', '/valid/valid_true/', '/valid/valid_predicted/')
            ]

    data_saves = [(X_test, y_test), (X_train, y_train), (X_valid, y_valid)]

    for x in range(len(results)):
        if(x == 0):
            if os.path.exists(results[x]):
                shutil.rmtree(results[x])
            os.makedirs(results[x])
        else:
            if os.path.exists(results[0]+"/"+results[x]):
                shutil.rmtree(results[0]+"/"+results[x])
            os.makedirs(results[0]+"/"+results[x])

    for i in range(3):
        for j in range(3):
            if os.path.exists(results[0]+dirs[i][j]):
                shutil.rmtree(results[0]+dirs[i][j])
            os.makedirs(results[0]+dirs[i][j])

            if(j == 0):
                for k in range(len(data_saves[i][0])):
                    plt.imsave(results[0]+dirs[i][j] +
                               str(k+1)+".jpeg", data_saves[i][0][k])
            if(j == 1):
                for k in range(len(data_saves[i][1])):
                    Image.fromarray((data_saves[i][1][k] * 255).astype(np.uint8).reshape(
                        256, 256)).save(results[0]+dirs[i][j]+str(k+1)+".jpeg")
            if(j == 2):
                pred = []
                if(model_number > 0):
                    pred = model_best.predict(data_saves[i][0])[0]
                else:
                    pred = model_best.predict(data_saves[i][0])
                for k in range(len(pred)):
                    Image.fromarray((pred[k] * 255).astype(np.uint8).reshape(
                        256, 256)).save(results[0]+dirs[i][j]+str(k+1)+".jpeg")


# In[1]:


def predict_classwise():

    category_ = ['train/', 'test/', 'validation/']
    tumor = ['benign/', 'malginant/', 'normal/']
    dirs_ = ['tumor/', 'true/', 'predicted/']

    path_x = "/workspace/classwise_results/"
    if os.path.exists(path_x):
        shutil.rmtree(path_x)
    os.makedirs(path_x)

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
                        pred = model_best.predict(X)[0]
                    else:
                        pred = model_best.predict(X)
                    for m in range(len(pred)):
                        Image.fromarray(
                            (pred[m] * 255).astype(np.uint8).reshape(256, 256)).save(path_ + str(m) + ".jpeg")
                    print(path_+" : ")
                    print(DiceAccuracy(np.asarray(l[i][j][1], dtype=np.float32)/255, pred).numpy())

# In[2]:


predict_data()
predict_classwise()


time.sleep(36000)
