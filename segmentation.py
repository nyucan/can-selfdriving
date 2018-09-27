# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob
import time

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import data_read
import util


image_h, image_w = (48, 160)
num_classes = 2
training_epochs = 3
batch_size = 10
learning_rate = 0.00005


def loss(y_true, y_pred):
    # flatten data
    logits_flat = tf.reshape(y_pred, (-1, num_classes))
    labels_flat = tf.reshape(y_true, (-1, num_classes))

    # Define loss
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_flat, logits=logits_flat))
    cross_entropy_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels_flat, logits=logits_flat, pos_weight=5))
    return cross_entropy_loss


def build_layers(model):
    model.add(keras.layers.Conv2D(input_shape=(48, 160, 3), filters=16, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=2, kernel_size=(1, 1), padding='same'))
    model.add(keras.layers.Softmax(axis=3))


def train(model, data, labels):
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=loss, metrics=['mae', 'acc'])
    model.fit(data, labels, epochs=training_epochs, callbacks=callbacks, batch_size=batch_size)


def save_checkpoint(path, model):
    t = time.time()
    keras.models.save_model(model, os.path.join(path, str(t)), overwrite=True, include_optimizer=True)


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
    data_folder = join(training_data_dir, 'aug-data')
    label_folder = join(training_data_dir, 'aug-label')

    # read dataset into memory
    data, labels = data_read.get_data(data_folder, label_folder, image_h, image_w)

    if from_model == None:
        print('info: Train FCNN from sketch')
        model = keras.Sequential()
        build_layers(model)
    else:
        model = from_model

    train(model, data, labels)
    save_checkpoint(join('.', 'models'), model)
    # result = model.predict(data, batch_size=1)
    # output(result, False)


def output(result, is_test, result_name=None):
    for i in range(len(result)):
        rgb_img = util.transfer_to_rgb(result[i])
        if is_test:
            save_path = join('.', 'data', 'output-test', result_name[i])
        else:
            save_path = join('.', 'data', 'output', str(i+1) + '.png')
        cv2.imwrite(save_path, rgb_img)


if __name__ == '__main__':
    pre_model = load('./models/1537812102.19')
    # run(pre_model)
    test()
