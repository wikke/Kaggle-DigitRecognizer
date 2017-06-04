# coding: utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, get_dummies

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

print (keras.__version__)

train = read_csv('train.csv')
test = read_csv('test.csv')

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

m4.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='SAME', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)))
m4.add(Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='SAME', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='SAME'))

m4.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='SAME', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)))
m4.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='SAME', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)))
m4.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='SAME'))

m4.add(Flatten())
m4.add(Dropout(0.3))
m4.add(Dense(128, activation='relu'))
m4.add(Dense(NUM_CLASSES, activation='softmax'))
m4.summary()
m4.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

MODEL_PATH = './model'
EPOCHS = 32
BATCH_SIZE = 128

tb4 = TensorBoard(log_dir=MODEL_PATH+'/m4/', write_images=True)
cp4 = ModelCheckpoint(filepath=MODEL_PATH+"/m4/checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

his4 = m4.fit(X_all, y_all, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, verbose=0, callbacks=[tb4, cp4, early_stopping])

y_pred = m4.predict_classes(X_test, batch_size=BATCH_SIZE)

np.savetxt('prediction.csv', np.c_[range(1, len(yPred)+1), y_pred], 
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')

