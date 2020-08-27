import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import keras.backend as K

def decode_tfrec_labeled(example):
    '''
    map function
    input：dataset slice
    output：image-np array（dim，dim，channel）
    '''
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'image_name': tf.io.FixedLenFeature([], tf.string),
                        'target': tf.io.FixedLenFeature([], tf.int64, default_value=0)}
    example = tf.io.parse_single_example(example, keys_to_features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    label = example['target']
    return image, label

def decode_tfrec_unlabeled(example):  # map function
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'image_name': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, keys_to_features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image_name = example['image_name']
    return image, image_name

def prepare_datasets(filenames, repeat=1, shuffle=True, label=True, augment=False, batch_size=32):
    '''
    convert ['str','str'...] to dataset
    :param filenames: ['str','str'...]
    :param repeat: TTA augment次数
    :param shuffle: train 必须 test不用
    :param label:
    :param augment:
    :param batch_size: [batch,[dim,dim,3]]的batch size
    :return: dataset-[image, label]
    '''
    AUTO = tf.data.experimental.AUTOTUNE
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.cache()
    ds = ds.repeat(repeat)

    if shuffle:
        ds = ds.shuffle(8000)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = ds.with_options(ignore_order)

    if label == True:
        ds = ds.map(decode_tfrec_labeled)
    else:
        ds = ds.map(decode_tfrec_unlabeled)

    if augment == True:
        ds = ds.map(lambda x, y: (augment_fun(x), y))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds