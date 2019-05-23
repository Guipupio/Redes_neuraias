#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:56:25 2019

@author: pupio
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#### Solucao Proposta ####
def solucao_inicial():
    model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model.evaluate(x_test, y_test)
    

def get_media(n=50):
    results=[]
    for i in range(n):
        results.append(solucao_inicial())
    
    return results

def get_calcula_media(results:list):
    loss = 0
    acc = 0
    for result in results:
        loss += result[0]
        acc += result[1]
    
    return {'loss': loss/len(results), 'acc':acc/len(results)}

#### Solucao De

def melhor_CNN():
    X_train = x_train.reshape(60000,28,28,1)
    X_test = x_test.reshape(10000,28,28,1)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64,kernel_size=10, activation='relu', strides=2, padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3,padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
#    model.add(tf.keras.layers.GlobalAveragePooling2D())
#    model.add(tf.keras.layers.Conv2D(15,kernel_size=5, activation='relu'))
#    model.add(tf.keras.layers.Conv2D(1,kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, use_multiprocessing=True, batch_size=None)
    model.evaluate(X_test, y_test, use_multiprocessing=True)

melhor_CNN()    

def desenvolvimento_com_CNN():
    X_train = x_train.reshape(60000,28,28,1)
    X_test = x_test.reshape(10000,28,28,1)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64,kernel_size=10, activation='relu', strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.22))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3,padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=1.5),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, use_multiprocessing=True, batch_size=None)
    model.evaluate(X_test, y_test, use_multiprocessing=True)

desenvolvimento_com_CNN()

def desenvolvimento_():
    X_train = x_train.reshape(60000,28,28,1)
    X_test = x_test.reshape(10000,28,28,1)
    model = tf.keras.models.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.elu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.nn.elu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=7, use_multiprocessing=True)
    return model.evaluate(X_test, y_test, use_multiprocessing=True)