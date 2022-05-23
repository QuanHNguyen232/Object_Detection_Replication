from tabnanny import verbose
from const import EPOCHS, BATCH_SIZE
import cv2
import tensorflow as tf
import numpy as np
import matplotlib as plt
from model import Yolov1
# from loss import yolo_loss
import utils
from dataset import get_data, get_1_img


def runModel():
    m = Yolov1((320, 320, 3), 5)
    model = m.getModel()
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    return model

if __name__ == '__main__':

    with tf.device("/GPU:0"):

        #======================#
        #       LOAD DATA      #
        #======================#
        # X, y, X_train, X_test, y_train, y_test = get_data(10)
                
        # val_steps =  X_train.shape[0] // BATCH_SIZE
        # print(f'val_steps: {val_steps}')
        
        #======================#
        #       RUN MODEL      #
        #======================#
        # yolo_model = runModel()
        # yolo_model.fit(X_train, y_train, 
        #                 epochs=EPOCHS,
        #                 batch_size=BATCH_SIZE,
        #                 validation_split = 0.2)
                        # validation_steps=val_steps)
        
        # yolo_model.save('whole-model.h5')
        

        #======================#
        #   LOAD SAVED MODEL   #
        #======================#
        # loaded_model = tf.keras.models.load_model('whole-model.h5')
        

        #======================#
        #       CHECK GPU      #
        #======================#
        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # c = tf.matmul(a, b)
        # print(c)
        print('done main.py')
