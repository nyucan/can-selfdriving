# python 2.7
from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob
import time

import numpy as np
import cv2

from fcn.fcn import Fcn
import data_read
import util


image_h, image_w = (48, 160)
training_data_dir = join('.', 'data', 'training')
data_folder = join(training_data_dir, 'aug-data')
label_folder = join(training_data_dir, 'aug-label')
testing_data_dir = join('.', 'data', 'testing', 'image_2')
output_test_dir = join('.', 'data', 'output-test')

checkpoint_path = join('.', 'models')
log_path = join('.', 'logs')
model_path = os.path.join('.', 'models')


def train_nn_from_sketch():
    data, labels = data_read.get_data(data_folder, label_folder, image_h, image_w)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.build_layers()
    nn.train()
    nn.save_checkpoint()
    return nn


def train_nn_from_model(model_name):
    data, labels = data_read.get_data(data_folder, label_folder, image_h, image_w)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.load_model(join(model_path, model_name))
    nn.train()
    nn.save_checkpoint()
    return nn


def test_model(nn):
    test_data, test_names = data_read.get_test_data(testing_data_dir, image_h, image_w)
    result = nn.predict(test_data)
    output(result, test_names)


def output(result, result_name=None):
    for i in range(len(result)):
        rgb_img = util.transfer_to_rgb(result[i])
        save_path = join(output_test_dir, result_name[i])
        cv2.imwrite(save_path, rgb_img)
    # util.put_images_to(rgb_imgaes, output_test_dir, result_name)


if __name__ == '__main__':
    # nn = train_nn_from_sketch()
    nn = train_nn_from_model('1538068014.76')
    test_model(nn)
