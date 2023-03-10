from __future__ import division, print_function, absolute_import
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
import Scalagram
from Scalagram import Scalagram
import matplotlib.pyplot as plt

from keras import backend as K 

def make_discriminator(input_shape):  
    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(32, 5, input_shape = input_shape, padding='same', activation='relu', name='conv1'),
        keras.layers.MaxPool2D(pool_size=2, padding='same', name='pool1'),
        keras.layers.Conv2D(64, 2, input_shape = input_shape, padding='same', activation='relu', name='conv2'),
        keras.layers.MaxPool2D(pool_size=2, padding='same', name='pool2'),
        keras.layers.Conv2D(96, 2, input_shape = input_shape, padding='same', activation='relu', name='conv3'),
        keras.layers.MaxPool2D(pool_size=2, padding='same', name='pool3'), 
        keras.layers.Conv2D(96, 2, input_shape = input_shape, padding='same', activation='relu', name='conv4'),
        keras.layers.MaxPool2D(pool_size=10, padding='same', name='pool4'),
        keras.layers.Lambda(lambda x: K.squeeze(x,2)),
        keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True), name='bi_lstm1'),
        keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True), name='bi_lstm2'),
        keras.layers.Dense(units=1, name='logits')
    ])
    discriminator.summary()
    return discriminator

def make_generator(input_shape):  
    generator = keras.models.Sequential([
        keras.layers.Conv2D(96, 5, input_shape = input_shape, padding='same', activation='relu', name='conv1'),
        keras.layers.Conv2D(96, 2, input_shape = input_shape, padding='same', activation='relu', name='conv2'),
        keras.layers.Conv2D(96, 2, input_shape = input_shape, padding='same', activation='relu', name='conv3'),
        keras.layers.Conv2D(96, 2, input_shape = input_shape, padding='same', activation='relu', name='conv4'),
        keras.layers.Lambda(lambda x: K.squeeze(x,2)),
        keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True), name='bi_lstm1'),
        keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True), name='bi_lstm2'),
        keras.layers.Reshape([28, 14,2])
    ])
    generator.summary()
    return generator
    
if __name__ == '__main__':
    file_location = "/Users/lucasgover/Desktop/Wavelet-Transform/Data/SwainsonCut.wav"
    sg = Scalagram(file_location)
    image = sg.get_data()
    print(image.shape)
    make_discriminator(image)