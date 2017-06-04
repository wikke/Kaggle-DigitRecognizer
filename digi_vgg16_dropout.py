# coding: utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, get_dummies

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

print('Keras version:' + keras.__version__)

train = read_csv('train.csv')
test = read_csv('test.csv')

print("Data read")

SAMPLE_SIZE = 42000
NUM_CLASSES = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNEL = 1

X_all = train.iloc[:SAMPLE_SIZE, 1:].copy()
y_all = train.iloc[:SAMPLE_SIZE, 0].copy()
X_test = test.copy()

X_all = X_all.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_all = X_all.values.reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)
X_test = X_test.values.reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)
y_all = get_dummies(y_all.values).values

m4 = Sequential()

m4.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='b1conv1', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)))
m4.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='b1conv2'))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME', name='b1pool'))

m4.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='b2conv1'))
m4.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='b2conv2'))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME', name='b2pool'))

m4.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='b3conv1'))
m4.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='b3conv2'))
m4.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='b3conv3'))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME', name='b3pool'))

m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b4conv1'))
m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b4conv2'))
m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b4conv3'))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME', name='b4pool'))

m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b5conv1'))
m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b5conv2'))
m4.add(Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='b5conv3'))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME', name='b5pool'))

m4.add(Flatten(name='flatten'))
m4.add(Dropout(0.3, name='dropout'))
m4.add(Dense(4096, activation='relu', name='fc1'))
m4.add(Dense(4096, activation='relu', name='fc2'))
m4.add(Dense(NUM_CLASSES, activation='softmax', name='softmax'))
m4.summary()
m4.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

print("Model Compiled")

MODEL_PATH = './model/vgg16dropout'
EPOCHS = 64 
BATCH_SIZE = 128

tb4 = TensorBoard(log_dir=MODEL_PATH, write_images=True)
cp4 = ModelCheckpoint(filepath=MODEL_PATH+"/checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

print("Start fitting")

his4 = m4.fit(X_all, y_all, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, verbose=0, callbacks=[tb4, cp4, early_stopping])

y_pred = m4.predict_classes(X_test, batch_size=BATCH_SIZE)

np.savetxt('prediction_vgg16_dropout.csv', np.c_[range(1, len(y_pred)+1), y_pred], 
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')
