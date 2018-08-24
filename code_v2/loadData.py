import numpy as np
import os
import imageio
from scipy.misc import imread
import cv2

# read all rgb images from folder
def readColorImage(subdir, resize = True):
    files = os.listdir(subdir)
    cm_1 = []
    cm_2 = []
    flag = 0
    for i, file in enumerate(sorted(files)):  # instantiates the colormatrix and timestamp color
        im = imageio.imread(subdir + file)
        if resize == True:
            im = cv2.resize(im, (1024, 256))
        if flag == 0:
            cm_1.append(im)
            flag = 1
        else:
            cm_2.append(im)
            flag = 0
    cm1 = np.array(cm_1)
    cm2 = np.array(cm_2)
    return cm1, cm2

def readColorImage_video(subdir, resize = True):

    cm = []
    folders= os.listdir(subdir)


    for i, folder in enumerate(sorted(folders)):


        for filename in os.listdir(subdir + folder + '/' ):
            dst = filename[8:]
        files = os.listdir(subdir + folder + '/')
        for j, file in enumerate(sorted(files)):  # instantiates the colormatrix and timestamp color
            im = imageio.imread(subdir + folder + '/' + file)
            if resize == True:
                im = cv2.resize(im, (512, 512))
            cm.append(im)
        cm = np.array(cm)
        print(i)
        np.save((str(i)+'.npy'), cm)
        cm = []
    return

def load(start, end):
    im1 = []
    im2 = []
    for i in range(start,end):
        arr = np.load(str(i)+'.npy')
        im1_arr = arr[:-1]
        im2_arr = arr[1:]
        im1.append(im1_arr)
        im2.append(im2_arr)
    im1 = np.concatenate(im1, 0)
    im2 = np.concatenate(im2, 0)
    return im1, im2

def load_chair(src = '/home/paris/Projects/FlyingChairs_release/data'):
    files = os.listdir(src)
    cm_1 = []
    cm_2 = []
    num = 0
    for i, file in enumerate(sorted(files)):
        if 'img1' in file:
            print(i)
            im = imread(src + '/' + file)
            cm_1.append(im)

        elif 'img2' in file:
            print(i)
            im = imread(src + '/' + file)
            cm_2.append(im)
        if i % 3000 == 0 and i > 1:
            cm_1 = np.array(cm_1)
            cm_2 = np.array(cm_2)
            np.save('./chairs/'+str(num)+'_cm_1.npy', cm_1)
            np.save('./chairs/'+str(num) + '_cm_2.npy', cm_2)
            cm_1 = []
            cm_2 = []
            num += 1