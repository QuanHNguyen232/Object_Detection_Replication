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
import dataset


class Yolov1(tf.keras.Model):   # Model based on YOLOv1 paper: https://arxiv.org/pdf/1506.02640.pdf
  def __init__(self, input_shape, n_out, n_class=0) -> None:
    super().__init__()
    self.n_class = n_class
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (7, 7), padding="same", input_shape=input_shape),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(192, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(128, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(256, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.MaxPool2D(2, 2),

      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2, 2),
      
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(512, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(1024, (3,3), strides=(2,2), padding="valid"),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      # tf.keras.layers.Dense(1 * 1 * n_out),
      # tf.keras.layers.Reshape((1, 1, n_out))

      tf.keras.layers.Dense(1)
    ])
    self.model.summary()
    

  def call(self, x):
    out = self.model(x)
    out_classes = tf.nn.sigmoid(out[..., :self.n_class])
    out_coords = out[..., self.n_class: self.n_class+4] # Need to check +4
    out_probs = tf.nn.sigmoid(out[..., self.n_class+4:])    # Need to check +4
    ret_val = tf.concat([out_classes, out_coords, out_probs], axis=-1)
    return ret_val
    # return out

  def getModel(self):
    return self.model
  def getSummary(self):
    return self.model.summary()

def test(input_shape=(320, 320, 3), n_anchor_boxes=9, n_out=5):
  m = Yolov1(input_shape, n_out, 7)
  model = m.getModel()
  x = tf.random.normal([16, 320, 320, 3])
  y_hat = model(x, training=True)
  print(f'y_hat.shape = {y_hat.shape}')




if __name__ == '__main__':
    test()
    print('done model.py')