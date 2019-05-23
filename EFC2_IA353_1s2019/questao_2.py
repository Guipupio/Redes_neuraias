#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:32:51 2019

@author: pupio
"""

import tensorflow as tf
import os

## DADOS 
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0



def solucao_inicial():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
     activation='relu',
    input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model.evaluate(x_test, y_test)


def melhor_CNN():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64,kernel_size=10, activation='relu', strides=2, padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3,padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10, use_multiprocessing=True, batch_size=None)
    return model.evaluate(x_test, y_test, use_multiprocessing=True)

melhor_CNN()

def desenvolvimento():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(1,kernel_size=3, activation='relu', strides=2, padding='same', dilation_rate=1))
    model.add(tf.keras.layers.Conv2D(60,kernel_size=10, activation='relu', strides=1, padding='same', dilation_rate=2))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3,padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=7, use_multiprocessing=True, batch_size=None)
    return model.evaluate(x_test, y_test, use_multiprocessing=True)

desenvolvimento()


file = open(r"C:\Users\Gabriel\.pupio\Redes_neuraias\EFC2_IA353_1s2019\aaa.txt","w+")
n = 20
file.writelines("[")
for i in range(n):
    print("Iteracao: " + str(i+1) + "\nProgresso: " + str(int(100*(1+i)/n)) +"%" )
    file.writelines(str(melhor_CNN())+",\n")

file.writelines("]")
file.close()
