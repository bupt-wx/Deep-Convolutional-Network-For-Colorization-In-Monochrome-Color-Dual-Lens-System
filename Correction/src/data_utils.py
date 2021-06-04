import numpy as np
import cv2
import os
import glob
import random
import math


def genDrivingPath(x, y):
        l_paths = []
        r_paths = []
        y_paths = []
        focal_lengths = ["15mm_focallength", "35mm_focallength"]
        directions = ["scene_backwards", "scene_forwards"]
        types = ["fast", "slow"]
        sides = ["left", "right"]
        for focal_length in focal_lengths:
                for direction in directions:
                        for type in types:
                                l_paths.append(os.path.join(x, *[focal_length, direction, type]))
                                r_paths.append(os.path.join(x, *[focal_length, direction, type]))
                                y_paths.append(os.path.join(y, *[focal_length, direction, type, sides[0]]))
        return l_paths, r_paths, y_paths

def genMonkaaPath(x, y):
        l_paths = []
        r_paths = []
        y_paths = []
        scenes = sorted(os.listdir(x))
        sides = ["left", "right"]
        for scene in scenes:
                        l_paths.append(os.path.join(x, *[scene, sides[0]]))
                        r_paths.append(os.path.join(x, *[scene, sides[1]]))
                        y_paths.append(os.path.join(y, *[scene, sides[0]]))
        return l_paths, r_paths, y_paths

def extractAllImage(lefts, rights):
        left_images = sorted(glob.glob(lefts + "/*.png"))
        right_images = sorted(glob.glob(rights + "/*.png"))
        return left_images, right_images

def splitData(l, r, val_ratio, fraction = 1):
        #tmp = zip(l, r)
        tmp = [(lhs, rhs) for lhs, rhs in zip(l, r)]
        random.shuffle(tmp)
        num_samples = len(l)
        num_data = int(fraction * num_samples)
        tmp = tmp[0:num_data]
        val_samples = int(math.ceil(num_data * val_ratio))
        val = tmp[0:val_samples]
        train = tmp[val_samples:]
        l_val, r_val = zip(*val)
        l_train, r_train = zip(*train)
        return [l_train, r_train], [l_val, r_val]