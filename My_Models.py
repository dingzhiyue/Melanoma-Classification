import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import keras.backend as K
import matplotlib.pyplot as plt

def focal_loss(gamma=2., alpha=.25):
    '''
    focal loss function
    '''

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def lrfn(epoch):#learning rate schedule function
    mult = 1
    lr_start = 6e-6
    lr_max = 1.45e-6 * 512
    lr_min = 1e-6
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.7
    if epoch < lr_ramp_ep:
        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
    elif epoch < lr_ramp_ep + lr_sus_ep:
        lr = lr_max
    else:
        lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
    return lr * mult


def model_effnet_flatten():
    '''
    flatten layer
    '''
    model = keras.models.Sequential()
    model.add(tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(384, 384, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(250, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    loss = focal_loss(2, 2.5)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer='Adam', loss=loss, metrics=[['AUC', 'accuracy']])
    return model

def model_effnet_averagepooling():
    '''
    GlobalAveragePooling2D layer
    '''
    model = keras.models.Sequential()
    model.add(tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(384, 384, 3)))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    # loss = focal_loss(2, 2.5)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer='Adam', loss=loss, metrics=[['AUC', 'accuracy']])
    return model