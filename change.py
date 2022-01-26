from PIL import Image
import os

def resize(fname,dirname):
    dirs = os.listdir(fname)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for item in dirs:
        if os.path.isfile(fname+item) and not item.endswith(").png"):

            im = Image.open(fname+item)
            f, e = os.path.splitext(fname+item)
            f=dirname+f.split('/')[-1]
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            
            imResize = im.resize((256,256), Image.ANTIALIAS).convert('L')
            imResize.save(f + ' resized_greyscale.jpg', 'JPEG', quality=90)

        elif os.path.isfile(fname+item) and item.endswith(".png"):

            im = Image.open(fname+item)
            f, e = os.path.splitext(fname+item)
            f=dirname+f.split('/')[-1]
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")

            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

file_dir="D:/tumor/Dataset_BUSI_with_GT/"  #original image directory
resize_dir="D:/tumor/resized/"             #resize storage directory
strs=os.listdir(file_dir)

for i in range(0,3):
    resize(file_dir+strs[i]+"/",resize_dir+strs[i]+"/")