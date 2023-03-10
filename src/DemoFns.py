from __future__ import division, print_function, absolute_import
import pyaudio
import wave
from csg import Chromoscalogram as Cscalogram
import numpy as np
from matplotlib import pyplot as plt
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
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wvf

def record(length:int):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = length
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def make_image(FILENAME):
    args = [FILENAME]
    cscal_files = [a for a in args if a.lower().endswith(".cscal")] + [a for a in args if a.lower().endswith(".wav")]


    cscals = []
    for name in cscal_files: cscals.append(Cscalogram(name))

    #ToDo: find largest amplitude among all cscalograms (scale) to lognormalize many
    scale= None 

    really_important_variable = []

    for i in cscals:
        really_important_variable += [i.write_to_color_png()]
        #plt.imshow(i.write_to_color_png(),interpolation ='nearest', origin ='lower',aspect='auto')
        #plt.show()
    plt.imshow(really_important_variable[0],interpolation ='nearest', origin ='lower',aspect='auto')
    plt.show()
        
def train_gan(gan, discriminator, dataset, batch_size, codings_size, n_epochs=20):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))              # not shown in the book
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_batch = tf.cast(X_batch, tf.float32)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            generated_images = generated_images.numpy().reshape(batch_size,28,28)
            plt.show()
from tensorflow import keras
from keras import layers


#img_shape=(frequencies, time, depth)
def build_discriminator(img_shape=(64, None, 2)):
    inputs = layers.Input(shape=img_shape, name='disc_input')
    
    #Convolution and downsampling until the 'frequency' axis is of length 1
    x = layers.Conv2D(32, (5,5), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool1')(x)
    x = layers.Conv2D(64, (5,5), padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool2')(x)
    x = layers.Conv2D(128, (5,5), padding='same', activation='relu', name='conv3')(x)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool3')(x)
    x = layers.Conv2D(256, (5,5), padding='same', activation='relu', name='conv4')(x)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool4')(x)
    x = layers.Conv2D(256, (5,5), padding='same', activation='relu', name='conv5')(x)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool5')(x)
    x = layers.Conv2D(256, (5,5), padding='same', activation='relu', name='conv6')(x)
    x = layers.MaxPool2D(pool_size=(2,1), padding='same', name='pool6')(x)

    #Let the channels combine into a single 2D shape, to feed to GRU layers over the 'time' axis
    x = layers.Reshape((-1, 256), name='reshape1')(x)
    
    
    x = layers.GRU(256, input_shape=(None,256), return_sequences=True, name='gru1')(x)
    x = layers.GRU(256, input_shape=(None,256), return_sequences=True, name='gru2')(x)
   
    #Reduce the 'time' axis to max value at each level
    x = tf.math.reduce_max(x, 1, name='pool5')


    x = layers.Dense(1, activation='sigmoid', name='activaton')(x)


    model = keras.Model(inputs=inputs, outputs=x)
    return model

def build_generator():
    inputs = layers.Input(shape=(None, 1), name='gen_input')
    
    #gives a depth of 256 to whatever length of input supplied
    x = layers.GRU(256, input_shape=(None,256), return_sequences=True, name='gru3')(inputs)
    x = layers.GRU(256, input_shape=(None,256), return_sequences=True, name='gru4')(x)
    x = layers.GRU(256, input_shape=(None,256), return_sequences=True, name='gru5')(x)

    x = layers.Reshape((1,-1, 256), name='reshape3')(x)

    x = layers.Conv2DTranspose(256, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv7')(x)
    x = layers.Conv2DTranspose(256, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv8')(x)
    x = layers.Conv2DTranspose(256, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv9')(x)
    x = layers.Conv2DTranspose(128, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv10')(x)
    x = layers.Conv2DTranspose(64, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv11')(x)
    x = layers.Conv2DTranspose(32, (5,5), strides=(2,1), padding='same', activation='relu', use_bias=False, name='conv12')(x)
    
    #Tanh for negative results (useful for magnitude, and probably phase)
    x = layers.Conv2DTranspose(2, (5,5), strides=(1,1), padding='same', activation='tanh', use_bias=False, name='conv13')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

def make_dataset(image):
    X = [image]
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    return dataset

def fwavelength(w_0):
    return 4 * np.pi / (w_0 + np.sqrt(2 + w_0**2))


def icwt(cwtm,scales,times):
    n_scales = np.shape(cwtm)[0]
    n_times = np.shape(cwtm)[1]

    diffs = np.empty(np.shape(cwtm),dtype=np.complex128)

    for i in range(n_scales):
        diffs[i] = approximate_diff(cwtm[i],times)


    f_rec = np.empty(n_times)

    for j in range(n_times):
        integral = trapezoid_int(diffs[:,j],scales)
        f_rec[j] = integral.imag / np.sqrt(2 * np.pi)

    return f_rec#.imag / np.sqrt(2 * np.pi)


def inv_transform_demo(FILENAME:str):
    hz = 200180
    hz, data = wvf.read(FILENAME)

    signal = data[:0] + data[:1]
    
    print(sigal.shape)
    """
    ## do the morlet transform
    cwt_list,scales_list = rustlets.cwt_morlet(samples,hz,8)

    scales = np.array(scales_list)
    freqs = 1 / (scales * fwavelength(2 * np.pi))
    cwtmat = np.array(cwt_list,dtype=np.complex64).reshape((len(scales),-1))[:,:len(times)]

    plt.plot(times,samples,label="original")
    plt.show()

    plt.pcolormesh(times,freqs, np.abs(cwtmat), cmap='viridis', shading='gouraud')
    plt.show()


    f_rec = icwt(cwtmat,scales,times)

    plt.plot(times,samples,label="original")
    plt.plot(times,f_rec * np.pi,label="recovered")
    plt.legend()
    plt.show()

    plt.plot(times,f_rec/samples,label="ratio")
    plt.legend()
    plt.show()
    """


if __name__ == '__main__': 
    generator = build_generator()
    generator.summary()
    discriminator = build_discriminator()
    discriminator.summary()



    #mess = tf.random.normal([1,64, 261, 2])
    #noise = tf.random.normal([261,1])
    #print(len(discriminator(generator(noise))))
    #print(len(generator(noise)))#.get_shape())
    #print(discriminator(mess).get_shape())