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
        # X, y, X_train, X_test, y_train, y_test = get_data(100)
                
        # val_steps =  X_train.shape[0] // BATCH_SIZE
        # print(f'val_steps: {val_steps}')
        
        #======================#
        #       RUN MODEL      #
        #======================#
        # yolo_model = runModel()
        # yolo_model.fit(X_train, y_train, 
        #                 epochs=EPOCHS,
        #                 batch_size=BATCH_SIZE,
        #                 validation_split = 0.2,
        #                 validation_steps=val_steps)
        
        # yolo_model.save('whole-model.h5')
        

        #======================#
        #   LOAD SAVED MODEL   #
        #======================#

        loaded_model = tf.keras.models.load_model('whole-model.h5')
        
        

        #======================#
        #    SHOW IMG W/ BOX   #
        #======================#
        for i in range(2):
            i += 10
            path = "../../PASCAL_VOC/images/0000" + str(i) + ".jpg"
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = utils.preprocessImage(img=img)
            img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
            print(img.shape)
            y_pred = loaded_model.predict(img)
            print(y_pred)

            utils.open_img(file_path=path, x=int(320*y_pred[0][1]), y=int(y_pred[0][2]*320), w=int(y_pred[0][3]*320), h=int(y_pred[0][4]*320))

        # %%
        # val = [i+10 for i in range(2)]
        # for x in val:
        #     img = np.asarray(X_train[x])
        #     loc = y_train[x]
        #     print(loc.shape)
        #     print(img.shape)
        #     print(img)
        #     x_, y_, w_, h_ = loc[1:]
        #     x_, w_ = x_*img.shape[1], w_*img.shape[1]
        #     y_, h_ = y_*img.shape[0], h_*img.shape[0]
        #     x_1 = x_ - (w_/2.0)
        #     y_1 = y_ - (h_/2.0)
        #     x_2 = x_ + (w_/2.0)
        #     y_2 = y_ + (h_/2.0)
        #     img = cv2.rectangle(img, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 0, 255), 2)
        #     cv2.imshow('',img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # print(X_train.shape)
        # print(X_train[1].shape)
        # print(y_train.shape)
        # y_train = np.asarray(y_train)
        # y_test = np.asarray(y_test)

        # my_model = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', input_shape=X_train[1].shape),
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        #     tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        #     tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(5, activation='softmax') #Output layer
        #     ])
        # # opt = tf.keras.optimizers.Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.09, epsilon=1e-8)
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
        # my_model.compile(loss="categorical_crossentropy", optimizer=opt)
        # history = my_model.fit(X_train, y_train, epochs=5, batch_size=5, validation_split = 0.2)#, validation_steps=val_steps)




        # img = cv2.imread("../../PASCAL_VOC/1-human-images-neg/000013.jpg", cv2.IMREAD_COLOR)
        # img = preprocessImage(img=img)
        # img = np.asarray(img, dtype=np.float32)
        # img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        # print(f'img.shape: {img.shape}')
        # y_pred = my_model.predict(img)
        # print(f'y_pred.shape: {y_pred.shape}')
        # print(f'y_pred: {y_pred}')

