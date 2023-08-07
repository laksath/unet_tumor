import os, os.path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

data=[([],[]),([],[]),([],[])]
count_temp=0
directory="../resized/"
subdir=["benign/","malignant/","normal/"]

def LoadData(imgPath,subd,images,masked):
    global count_temp
    count_temp+=1
    
    if(count_temp>3):
        for i in range(3):
            data[i][0].clear()
            data[i][1].clear()
        count_temp=1
        
    imgPath=imgPath+subd
    imgNames = sorted(os.listdir(imgPath))

    for item in imgNames:
        img = plt.imread(imgPath + item)
        if item.endswith(')_mask resized_greyscale.jpg'):
            masked.append(img)
        elif item.endswith(') resized.jpg'):
            # print(item)
            images.append(img)
        elif item.endswith(' resized_greyscale.jpg'):
            masked[-1]=np.add(masked[-1], img)

def LoadData_Info():
    for i in range(3):
        LoadData(directory,subdir[i],i,data[i][0],data[i][1])


LoadData_Info()

dir = 'benign_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'benign_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[0][0])):
    im = Image.fromarray(data[0][0][i])
    im.save("benign_image/"+str(i)+".jpeg")

for i in range(len(data[0][0])):
    im = Image.fromarray(data[0][1][i])
    im.save("benign_mask/"+str(i)+".jpeg")

dir = 'malignant_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'malignant_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[1][0])):
    im = Image.fromarray(data[1][0][i])
    im.save("malignant_image/"+str(i)+".jpeg")

for i in range(len(data[1][0])):
    im = Image.fromarray(data[1][1][i])
    im.save("malignant_mask/"+str(i)+".jpeg")

dir = 'normal_image'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = 'normal_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

for i in range(len(data[2][0])):
    im = Image.fromarray(data[2][0][i])
    im.save("normal_image/"+str(i)+".jpeg")

for i in range(len(data[2][0])):
    im = Image.fromarray(data[2][1][i])
    im.save("normal_mask/"+str(i)+".jpeg")

