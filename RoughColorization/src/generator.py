import cv2
import numpy as np
import glob
import random
from load_pfm import *
import tensorflow as tf

def generate_arrays_from_file(lefts, rights, up, is_train):

        train = is_train # True or False
        random.seed(up['seed'])

        while 1:
            for ldata, rdata in zip(lefts, rights):
                left_img = cv2.imread(ldata)
                right_img = cv2.imread(rdata)

                left_img = _centerImage_(left_img)
                right_img = _centerImage_(right_img)
                left_img = np.expand_dims(left_img, 0)
                right_img = np.expand_dims(right_img, 0)
                
                if train == True:
                    VUY_map=left_img[:]
                    yield ([left_img, right_img], VUY_map)
                else:
                    yield ([left_img, right_img])
            if not train:
                break

def _centerImage_(img):
    img = img.astype(np.float32)
    return img

