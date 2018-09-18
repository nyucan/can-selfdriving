# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


smooth = 1
num_classes = 3


def get_data(data_folder, image_h, image_w):
    background_color = np.array([255, 255, 255]) # white
    left_lane_color = np.array([255, 0, 0])      # red
    right_lane_color = np.array([0, 0, 255])     # blue

    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = glob(os.path.join(data_folder, 'gt_image', '*.png'))

    # make sure the label and image are matched
    image_paths.sort()
    label_paths.sort()

    images = []    # data
    gt_images = [] # labels

    for image_file_id in range(0, len(image_paths)):
        image_file = image_paths[image_file_id]
        gt_image_file = label_paths[image_file_id]

        image = scipy.misc.imread(image_file, mode='RGB')
        gt_image = scipy.misc.imread(gt_image_file, mode='RGB')

        gt_bg = np.all(gt_image == background_color, axis=2).reshape(image_h, image_w, 1)
        gt_ll = np.all(gt_image == left_lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_rl = np.all(gt_image == right_lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_image = np.concatenate((gt_bg, gt_ll, gt_rl), axis=2)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)


def loss(y_true, y_pred):
    # flatten data
    logits_flat = tf.reshape(y_pred, (-1, num_classes))
    labels_flat = tf.reshape(y_true, (-1, num_classes))

    # Define loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_flat, logits=logits_flat))
    return cross_entropy_loss


def build_layers(model):
    model.add(keras.layers.Conv2D(input_shape=(48, 160, 3), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Softmax(axis=3))


def train(model, data, labels):
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=loss, metrics=['accuracy'])
    model.fit(data, labels, epochs=3, batch_size=10)


def evaluate(model, val_data, val_label):
    model.evaluate(val_data, val_label, batch_size=32)


def run():
    image_h, image_w = (48, 160)
    data_dir = join('.', 'data')
    training_data_dir = join(data_dir, 'training')

    data, labels = get_data(training_data_dir, image_h, image_w)
    model = keras.Sequential()

    build_layers(model)

    train(model, data, labels)
    # evaluate(model, data, labels)
    result = model.predict(data, batch_size=1)
    output(result)


def transfer_to_rgb(bin_img):
    h, w, c = bin_img.shape
    rgb_img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            if np.all(bin_img[i][j] == [1., 0., 0.]):
                rgb_img[i][j] = [255, 255, 255]
            elif np.all(bin_img[i][j] == [0., 1., 0.]):
                rgb_img[i][j] = [255, 0, 0]
            elif np.all(bin_img[i][j] == [0., 0., 1.]):
                rgb_img[i][j] = [0, 0, 255]
    return rgb_img


def output(result):
    for i in range(len(result)):
        rgb_img = transfer_to_rgb(result[i])
        save_path = os.path.join('.', 'data', 'output', str(i+1) + '.png')
        scipy.misc.imsave(save_path, rgb_img)


if __name__ == '__main__':
    run()
