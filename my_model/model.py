import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import json
import cv2
import os
import filter_img

# CONFIG
H, W = 500, 500
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_map = {k:idx for idx,k in enumerate(classes)}
nclasses = 0    # len(class_map)
output_shape = [5 + nclasses]

# DATA PREPARE
PATH = '../../PASCAL_VOC/'
labels_neg = filter_img.read_label(PATH + 'human-label-neg/')
labels_pos = filter_img.read_label(PATH + 'human-label-pos/')
labels = labels_neg + labels_pos
np.random.shuffle(labels)

train_label = labels[:50]
val_label = labels[50:]

batch_size = 128
train_steps = len(train_label) // batch_size
val_steps = len(val_label) // batch_size

imgs = filter_img.get_images(PATH + 'human-images-pos/')
print(imgs[3].shape)

if __name__ == '__main__':
    pass