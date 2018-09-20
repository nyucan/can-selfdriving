# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob
import time

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import data_read
import data_process


smooth = 1
num_classes = 3
training_epochs = 7
batch_size = 10
image_h, image_w = (48, 160)


def loss(y_true, y_pred):
    # flatten data
    logits_flat = tf.reshape(y_pred, (-1, num_classes))
    labels_flat = tf.reshape(y_true, (-1, num_classes))

    # Define loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_flat, logits=logits_flat))
    return cross_entropy_loss


def build_layers(model):
    model.add(keras.layers.Conv2D(input_shape=(48, 160, 3), filters=16, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same'))
    model.add(keras.layers.Softmax(axis=3))


def train(model, data, labels):
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=loss, metrics=['accuracy'])
    model.fit(data, labels, epochs=training_epochs, batch_size=batch_size)


def load(model_path):
    model = keras.models.load_model(model_path, custom_objects=None, compile=True)
    return model


def test():
    testing_data_dir = join('.', 'data', 'testing', 'image_2')
    test_data, test_names = data_read.get_test_data(testing_data_dir, image_h, image_w)
    model_path = os.path.join('.', 'models', 'test')
    model = load(model_path)
    result = model.predict(test_data, batch_size=1)
    output(result, True, test_names)


def run(from_model=None):
    image_h, image_w = (48, 160)
    training_data_dir = join('.', 'data', 'training')
    data_folder = join(training_data_dir, 'image_2')
    label_folder = join(training_data_dir, 'gt_image_2')

    # read dataset into memory
    data, labels = data_read.get_data(data_folder, label_folder, image_h, image_w)

    if from_model == None:
        model = keras.Sequential()
        print('info: Train FCNN from sketch')
        build_layers(model)
    else:
        model = from_model

    train(model, data, labels)
    keras.models.save_model(model, os.path.join('.', 'models', 'test'), overwrite=True, include_optimizer=True)

    result = model.predict(data, batch_size=1)
    output(result, False)


def output(result, is_test, result_name=None):
    for i in range(len(result)):
        rgb_img = data_process.transfer_to_rgb(result[i])
        if is_test:
            save_path = os.path.join('.', 'data', 'output-test', result_name[i])
        else:
            save_path = os.path.join('.', 'data', 'output', str(i+1) + '.png')
        scipy.misc.imsave(save_path, rgb_img)


if __name__ == '__main__':
    run()
    # test()
