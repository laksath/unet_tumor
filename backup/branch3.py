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

epochs_ = 100
batch_size_ = 25
model_number = 1
filepath = '/workspace/data/code_h/m1a0e100b25/'
# In[283]:


# In[284]:


########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################


# In[285]:


subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/code_h/dataset/',
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

def model_build(input_shape):
    inputs = Input(input_shape)
    a = Flatten()(inputs)
    b = Dense(1500, activation='relu')(a)
    y3_output = Dense(3,activation='softmax',  name='classifier')(b)
    model = Model(inputs=inputs, outputs=y3_output, name="U-Net")
    return model

improvised_unet_model = model_build((256,256))
improvised_unet_model.summary()

improvised_unet_model.compile(optimizer='adam',
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy']
                              )

improvised_unet_model.fit(y_train, y_train2, epochs=15)