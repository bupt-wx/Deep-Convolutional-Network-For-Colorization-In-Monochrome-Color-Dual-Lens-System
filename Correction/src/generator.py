import cv2
import numpy as np
import glob
import random
from load_pfm import *

def generate_arrays_from_file(roughs, guidances, up, is_train):

        train = is_train # True or False
        random.seed(up['seed'])

        while 1:
            for rough_data, guidance_data in zip(roughs, guidances):
                rough_img = cv2.imread(rough_data)
                rough_VU = rough_img[:,:,0:2]

                guidance_img = cv2.imread(guidance_data) 
                guidance_Y = guidance_img[:,:,2:3]
                guidance_VU = guidance_img[:,:,0:2]

                rough_VU = _centerImage_(rough_VU)
                guidance_Y = _centerImage_(guidance_Y)
                guidance_VU = _centerImage_(guidance_VU)

                rough_VU = np.expand_dims(rough_VU, 0)
                guidance_Y = np.expand_dims(guidance_Y, 0)
                guidance_VU = np.expand_dims(guidance_VU, 0)
                
                if train == True:
                    VU_map=guidance_VU[:]
                    yield ([rough_VU, guidance_Y], VU_map)
                else:
                    yield ([rough_VU, guidance_Y])
            if not train:
                break

def _centerImage_(img):
    img = img.astype(np.float32)
    return img
