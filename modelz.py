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
from numpy.random import seed
from tensorflow.keras.utils import to_categorical
from collections import Counter
import gc

epochs_ = 500
batch_size_ = 32
# filepath = '/workspace/data/code_j/m3a1e1000b32seed5_baseunet_1/'
# histpath = '/workspace/data/code_j/m3a1e1000b32seed5_baseunet_1.xlsx'



def setseed(seedn):
    seed(seedn)
    tf.random.set_seed(seedn)
    tf.keras.utils.set_random_seed(seedn)
    os.environ['PYTHONHASHSEED'] = str(seedn)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# setseed(5)


# weights = [1/3,1/3,1/3] 
#weights = [1/2,1/4,1/4]
#weights = [1/4,1/2,1/4]
#weights = [1/4,1/4,1/2]
# weights = [1,1,1]

weights = [0.5,0.5,1]



#unet variants :  https://arxiv.org/ftp/arxiv/papers/2011/2011.01118.pdf

# In[284]:

########################################################################################################################
####################################           KeyVars           #######################################################
########################################################################################################################

subdir = ["benign_image/", "benign_mask/", "malignant_image/",
          "malignant_mask/", "normal_image/", "normal_mask/"]
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

# In[127]:

########################################################################################################################
####################################         UNET MODEL          #######################################################
########################################################################################################################



def custom(custom_layer):
    inp = Input(shape=(256,256,3))
    l1 = custom_layer(inp)
    l2 = Flatten()(l1)
    l3 = Dense(1024, activation='relu')(l2)
    l4 = Dense(3)(l3)
    output = tf.keras.layers.Activation('softmax', name='classifier')(l4)
    model = Model(inputs=inp, outputs=output)
    model.summary()
    return model

layers=[
    tf.keras.applications.efficientnet.EfficientNetB2(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.efficientnet.EfficientNetB4(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.efficientnet.EfficientNetB6(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.resnet.ResNet101(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.resnet.ResNet152(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.densenet.DenseNet169(
        include_top=False,
        weights=None,
    ),
    tf.keras.applications.densenet.DenseNet201(
        include_top=False,
        weights=None,
    ),
]

model_string=[
    'EfficientNetB2',
    'EfficientNetB4',
    'EfficientNetB6',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
]



########################################################################################################################
####################################        LOSS FUNCTIONS         #####################################################
########################################################################################################################

def cce_loss(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_true,y_pred)

for i in range(9):
    if os.path.exists('/workspace/data/op2/'+model_string[i]):
        shutil.rmtree('/workspace/data/op2/'+model_string[i])
    os.makedirs('/workspace/data/op2/'+model_string[i])
    eff = custom(layers[i])
    filepath = '/workspace/data/op2/'+model_string[i]+'/'+model_string[i]+'/'
    histpath = '/workspace/data/op2/'+model_string[i]+'/'+model_string[i]+'.xlsx'
    eff.compile(
        optimizer='adam',
        loss=cce_loss,
        metrics=['accuracy'],
        )
    callbacks = [
        ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True),
    ]
    history = eff.fit(X_train, y_train2, batch_size=batch_size_,
                                        epochs=epochs_, validation_data=(X_valid, y_valid2), callbacks=callbacks)
    hist = pd.DataFrame(history.history)
    hist.to_excel(histpath)
    
# for i in [6]:
#     setseed(i)
    
#     if os.path.exists('/workspace/data/op2/m3_original_a1e1000b32b13seed'+str(i)):
#         shutil.rmtree('/workspace/data/op2/m3_original_a1e1000b32b13seed'+str(i))
#     os.makedirs('/workspace/data/op2/m3_original_a1e1000b32b13seed'+str(i))

#     original_improvised_unet_model = original_improvised_unet_build()
#     filepath = '/workspace/data/op2/m3_original_a1e1000b32b13seed'+str(i)+'/m3_original_a1e1000b32b12_final_seed'+str(i)+'/'
#     histpath = '/workspace/data/op2/m3_original_a1e1000b32b13seed'+str(i)+'/m3_original_a1e1000b32b12_final_seed'+str(i)+'.xlsx'

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
#     history = original_improvised_unet_model.fit(X_train, (y_train, X_train), batch_size=batch_size_,
#                                         epochs=epochs_, validation_data=(X_valid, (y_valid, X_valid)), callbacks=callbacks)

#     del original_improvised_unet_model
#     keras.clear_session()
#     gc.collect()

#     hist = pd.DataFrame(history.history)
#     hist.to_excel(histpath)

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