import sys
sys.path.append('src')
import numpy as np
import argparse
import parse_arguments
from coloringnetwork import *
import glob
import os
import psutil
from generator import *
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def Predict():
    hp, tp, up, env = parse_arguments.parseArguments()

    parser = argparse.ArgumentParser()
    parser.add_argument('-fpath', help = 'file path of the dataset', required = True, default = None)
    parser.add_argument('-outpath', help = 'file output path of the dataset', required = True, default = None)
    parser.add_argument('-bs', help = 'batch size or steps', default = tp['batch_size'])
    args = parser.parse_args()

    outpath = args.outpath
    ext = up['file_extension']
    bs = tp['batch_size']
    file_path = args.fpath
    max_q_size = tp['max_q_size']
    verbose = tp['verbose']

    def get_session(gpu_fraction=0.95):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session())
    model = createColoringNetwork(hp, tp, True)

    left_path = file_path + "/color_0_ycbcr"
    right_path = file_path + "/color_1_ycbcr"
    left_images = glob.glob(left_path + "/*.{}".format(ext))
    right_images = glob.glob(right_path + "/*.{}".format(ext))
    
    data_path_test_out = outpath + "/"

    list_len=len(left_images)
    print('left_images length',list_len)

    for i in range(0, list_len):
        print(i,left_images[i])
        left_image=[left_images[i]]
        right_image=[right_images[i]]

        print('left image length',len(left_image))
        is_train = False
        generator = generate_arrays_from_file(left_image, right_image, up, is_train)
        print('generator is',generator)
        print("Predict data using generator...")
        pred = model.predict_generator(generator, max_queue_size = max_q_size, steps = bs, verbose = verbose)
        pred = pred[0,:,:,:]

        imgage = cv2.imread(left_images[i])
        pred[:, :, 2:3] = imgage[:, :, 2:3]

        print(left_image[0].split('/')[-1].split('.')[0])

        cur_png_path = data_path_test_out + left_image[0].split('/')[-1]
        print(cur_png_path)

        f=cv2.imwrite(cur_png_path, pred)

    print("Testing Complete")
    K.clear_session()

if __name__ == "__main__":
    Predict()   
