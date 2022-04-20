import const
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import numpy as np
import matplotlib as plt
from model import Yolov1
from loss import yolo_loss
import utils
from dataset import get_data, preprocessImage

def runModel():
    
    m = Yolov1((320, 320, 3), 5)
    model = m.getModel()
    model.compile(loss=yolo_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
    return model

if __name__ == '__main__':
    with tf.device("/CPU:0"):

        X, y = get_data(100)
        y = y[..., :1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # val_steps =  len(X_train)// const.BATCH_SIZE
        # model = runModel()
        # model.fit(X_train, y_train, epochs=const.EPOCHS, batch_size=const.BATCH_SIZE, validation_split = 0.2)#, validation_steps=val_steps)
        


        drop_rate = 0.2
        my_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', input_shape=X_train[1].shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

            tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

            tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(1, activation='softmax') #Output layer
            ])
        opt = tf.keras.optimizers.Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.09, epsilon=1e-8)
        my_model.compile(loss="categorical_crossentropy", optimizer=opt)
        history = my_model.fit(X_train, y_train, epochs=const.EPOCHS, batch_size=const.BATCH_SIZE, validation_split = 0.2)#, validation_steps=val_steps)

        results_test = my_model.evaluate(X_test, y_test, batch_size=const.BATCH_SIZE,verbose=0)    
        my_model(history, results_test)
        plt.show()

        img = cv2.imread('imgs/hiking.jpg', cv2.IMREAD_COLOR)
        img = preprocessImage(img=img)
        img = np.asarray(img, dtype=np.float32)
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

        print(f'img.shape: {img.shape}')
        y_pred = my_model.predict(img)
        print(f'y_pred.shape: {y_pred.shape}')
        print(f'y_pred: {y_pred}')
        
        
        img = cv2.imread("../../PASCAL_VOC/1-human-images-neg/000013.jpg", cv2.IMREAD_COLOR)
        img = preprocessImage(img=img)
        img = np.asarray(img, dtype=np.float32)
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

        print(f'img.shape: {img.shape}')
        y_pred = my_model.predict(img)
        print(f'y_pred.shape: {y_pred.shape}')
        print(f'y_pred: {y_pred}')

def plot_result_4(history,results_test):
    # Get training and validation histories
    training_acc = history.history['accuracy']
    val_acc      = history.history['val_accuracy']
    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)
    # Visualize loss history
    plt.plot(epoch_count, training_acc, 'b-o',label='Training')
    plt.plot(epoch_count, val_acc, 'r--',label='Validation')
    plt.plot(epoch_count, results_test[1]*np.ones(len(epoch_count)),'k--',label='Test')
    plt.legend()
    plt.title("Training and validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')