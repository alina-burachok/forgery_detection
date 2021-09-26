from PIL import Image, ImageOps
import os
import sys
import string
import glob
from constants import *
from StringIO import StringIO

def cropCenter(im, desire_x, desire_y):
    try:
        imagewidth, imageheight = im.size
        # determine the dimensions of a crop area
        cropwidth = 0
        cropheight = 0
        
        # if the image is bigger than the desired size compute the size to crop
        if imagewidth > desire_x:
            cropwidth = int((imagewidth - desire_x) / 2)
            
        if imageheight > desire_y:
            cropheight = int((imageheight - desire_y) / 2)
        
        # computing  the pixels of the right corner of the box
        dx,dy = (imagewidth-cropwidth),(imageheight-cropheight)

        box = (cropwidth,cropheight,dx,dy)
        ima = im.crop(box)
        # calling loas method because crop is a lazy operation 
        # and changes to the source image may or may not be reflected in the cropped image.
        ima.load()
    except Exception, e:
        print "Exception: ", e
        ima = None
        pass
    return ima

def cropCorner(im, desire_x, desire_y):
    try:
        imagewidth, imageheight = im.size

        # if the image is bigger than the desired size compute the size to crop
        if imagewidth < desire_x:
            desire_x = imagewidth
            
        if imageheight < desire_y:
            desire_y = imageheight
    
        box = (0,0,desire_x,desire_y)
        ima = im.crop(box)
        # calling loas method because crop is a lazy operation 
        # and changes to the source image may or may not be reflected in the cropped image.
        ima.load()
    except Exception, e:
        print "Exception: ", e
        ima = None
        pass
    return ima
    
def cropImages(path, desire_x, desire_y, center):
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in EXTS:
            image = Image.open(os.path.join(path, f))
            if center:
                image_cropped = cropCenter(image, desire_x, desire_y)
            else:
                image_cropped = cropCorner(image, desire_x, desire_y)
            image_cropped.save(os.path.join(path, CROPPED_PREFIX + f))
    #print "--- Crop Finalized ---"

def cropImage(path, name, desire_x, desire_y, center):
    image = Image.open(os.path.join(path,name))
    if center:
        image_cropped = cropCenter(image, desire_x, desire_y)
    else:
        image_cropped = cropCorner(image, desire_x, desire_y)

    if SAVEIMGS_REQUIRED:
        image_cropped.save(os.path.join(path,CROPPED_PREFIX + name))
    return image_cropped

def cropImageFilename(filename, desire_x, desire_y, center):
    im = Image.open(filename)
    if center:
        image_cropped = cropCenter(im, desire_x, desire_y)
    else:
        image_cropped = cropCorner(im, desire_x, desire_y)
    return image_cropped

#crop imageBLOB and return PIL.Image
def cropImageBLOB(blob, desire_x, desire_y, center):
    image = Image.open(StringIO(blob))
    if center:
        image_cropped = cropCenter(image, desire_x, desire_y)
    else:
        image_cropped = cropCorner(image, desire_x, desire_y)
        
    return image_cropped
        
#cropImage('./','image.JPG', 256, 256)
#cropImage('C:\Documents and Settings\joss','joss.jpg', 300, 300, True)
#cropImages('./',CROP_X_SIZE, CROP_Y_SIZE, True)
