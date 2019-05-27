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
 

def desenvolvimento_(num_epochs, units_camada_um, dropout_um, camada_dois, dropout_dois, gradient):
    X_train = x_train.reshape(60000,28,28,1)
    X_test = x_test.reshape(10000,28,28,1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units_camada_um, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout_um))
    model.add(tf.keras.layers.Dense(camada_dois, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout_dois))
    
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.train.GradientDescentOptimizer(gradient),
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, use_multiprocessing=True)
    return model.evaluate(X_test, y_test, use_multiprocessing=True)

desenvolvimento_(num_epochs=11)

num_file = 2
n = 10
for num_epoch in [10, 11, 12]:
    for camada_um in [512, 800]:
        for dropout_um in [0.2,0.3,0.4]:
            for camada_dois in [32,64,128]:
                for dropout_dois in [0.2,0.3,0.4]:
                    for gradient in [0.17,0.18]:
                        file = file = open(r"C:\Users\Gabriel\.pupio\Redes_neuraias\EFC2_IA353_1s2019\tentativa_erro_"+ str(num_file) +".txt","w+")
                        file.writelines("Parametros: num_epoch=" + str(num_epoch) + "; camada_um="+ str(camada_um) + "; dropout_um="+ str(dropout_um) + "; camada_dois="+ str(camada_dois) + "; dropout_dois="+ str(dropout_dois) + "; gradient=" + str(gradient) + "\n[")
                        for i in range(n):
                            print("Iteracao: " + str(i+1) + "\nProgresso: " + str(int(100*(1+i)/n)) +"%" )
                            file.writelines(str(desenvolvimento_(num_epochs=num_epoch, units_camada_um=camada_um, dropout_um=dropout_um, camada_dois=camada_dois, dropout_dois=dropout_dois, gradient=gradient))+",\n")
                        
                        file.writelines("]")
                        file.close()
                        num_file += 1