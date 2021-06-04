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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def Predict():
    hp, tp, up, env = parse_arguments.parseArguments()

    parser = argparse.ArgumentParser()
    parser.add_argument('-rpath', help = 'path of fine images', required = True, default = None)
    parser.add_argument('-gpath', help = 'path of fine images', required = True, default = None)
    parser.add_argument('-outpath', help = 'file output path of the dataset', required = True, default = None)
    parser.add_argument('-bs', help = 'batch size or steps', default = tp['batch_size'])
    args = parser.parse_args()

    outpath = args.outpath
    ext = up['file_extension']
    bs = tp['batch_size']
    rough_path = args.rpath
    guidance_path = args.gpath
    max_q_size = tp['max_q_size']
    verbose = tp['verbose']

    def get_session(gpu_fraction=0.95):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session())
    model = createColoringNetwork(hp, tp, True)

    rough_images = glob.glob(rough_path + "/*.{}".format(ext))
    guidance_images = glob.glob(guidance_path + "/*.{}".format(ext))
    
    data_path_test_out = outpath + "/"

    list_len=len(rough_images)
    print('rough_images length',list_len)

    for i in range(0, list_len):
        print(i,rough_images[i])
        rough_image=[rough_images[i]]
        guidance_image=[guidance_images[i]]

        print('left image length',len(rough_image))
        is_train = False
        generator = generate_arrays_from_file(rough_image, guidance_image, up, is_train)
        print('generator is',generator)
        print("Predict data using generator...")

        pred = model.predict_generator(generator, max_queue_size = max_q_size, steps = bs, verbose = verbose)
        pred = pred[0,:,:,:]

        rough_image_val = cv2.imread(rough_image[0])

        output_image = rough_image_val
        output_image[:,:,0:2] = pred[:,:,0:2]

        print(rough_image[0].split('/')[-1].split('.')[0])
        
        cur_png_path = data_path_test_out + rough_image[0].split('/')[-1]
        print(cur_png_path)

        f=cv2.imwrite(cur_png_path, output_image)

    print("Testing Complete")
    K.clear_session()

if __name__ == "__main__":
    Predict()   
