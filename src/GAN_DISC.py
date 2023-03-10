import numpy as np
import tensorflow as tf
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
