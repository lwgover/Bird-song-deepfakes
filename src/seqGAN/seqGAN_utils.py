import tensorflow as tf
import pickle
from collections import Counter, defaultdict
from unicodedata import normalize
import re
import numpy as np
import keras.backend as K
import os
from keras.models import load_model
import math
from csg import Chromoscalogram

def load_data(file):
    sg = Chromoscalogram(file)
    image = sg.give_data_to_tf()
    dataset = []
    for i in range(100):
        dataset.append(image);
    return dataset



def loss_func(y_train, pred):
    mask = K.cast(y_train > 0, dtype='float32')
    mask2 = tf.greater(y_train, 0)
    non_zero_y = tf.boolean_mask(pred, mask2)
    val = K.log(non_zero_y)

    return  -K.sum(val) / K.sum(mask)


def acc_func(y_train, pred):
    targ = K.argmax(y_train, axis=-1)
    pred = K.argmax(pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    mask = K.cast(K.greater(targ, 0), dtype='float32')  # filter out padding value 0.
    correctCount = K.sum(mask * correct)
    totalCount = K.sum(mask)
    return  correctCount / totalCount

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def make_batch(X, Y, shuffle=True, batch_size=64):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    dataset = []
    batchs = math.ceil(len(X) / batch_size)
    for b in range(batchs):
        s = b * 64
        e = min(s + 64, len(X))
        dataset.append([b, X[s:e], Y[s:e]])
    return dataset

def save_result(loss, acc, loss_test, acc_test, best_acc, dir):
  f ={}
  f['loss'] = loss
  f['acc'] = acc
  f['loss_test'] = loss_test
  f['acc_test'] = acc_test
  f['best_acc'] = best_acc
  name = open(os.path.join(dir,'result.pkl'),'wb')
  pickle.dump(f,name)
  name.close()

def load_result(dir):
    pkl_file = open(os.path.join(dir, 'result.pkl'), 'rb')
    f = pickle.load(pkl_file)
    loss = f['loss']
    acc = f['acc']
    loss_test = f['loss_test']
    acc_test = f['acc_test']
    best_acc = f['best_acc']
    pkl_file.close()
    return loss, acc, loss_test, acc_test, best_acc


def save_model(model, epoch, dir):
  f = {}
  f['model'] = model
  f['epoch'] = epoch
  name = open(os.path.join(dir,'model.pkl'),'wb')
  pickle.dump(f,name)
  name.close()
  model.save(os.path.join(dir,'train_model.h5'), os.path.join(dir,'infer_model.h5'))



def load_model(dir):
  pkl_file = open(os.path.join(dir,'model.pkl'), 'rb')
  f = pickle.load(pkl_file)
  model = f['model']
  epoch = f['epoch']
  pkl_file.close()
  model.train_model = load_model(os.path.join(dir,'train_model.h5'))
  model.inference_model  = load_model(os.path.join(dir,'infer_model.h5'))
  return model, epoch