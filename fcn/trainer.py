# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import join, expanduser
from glob import glob
import time
import numpy as np
import cv2

from fcn import Fcn
from util import util

import dataset


IMAGE_H, IMAGE_W = (48, 160)
TRAINING_DATA_DIR = join('.', 'data', 'training')
DATA_DIR = join(TRAINING_DATA_DIR, 'aug-data')
LABEL_DIR = join(TRAINING_DATA_DIR, 'aug-label')
TESTING_DATA_DIR = join('.', 'data', 'testing', 'image')
TESTING_PRED_DIR = join('.', 'data', 'testing', 'predict')

CHECKPOINT_PATH = join('.', 'models')
LOG_PATH = join('.', 'logs')

def Trainer(object):
    def __init__(self):
        pass

    def train_nn_from_sketch(self, norm=False):
        data, labels = dataset.get_data(DATA_DIR, LABEL_DIR, IMAGE_H, IMAGE_W, norm)
        nn = Fcn(data, labels, (IMAGE_H, IMAGE_W), CHECKPOINT_PATH, LOG_PATH)
        nn.define_loss()
        nn.build_layers()
        nn.train()
        nn.save_checkpoint()
        return nn

    def train_nn_from_model(self, model_name, norm=False):
        data, labels = dataset.get_data(DATA_DIR, LABEL_DIR, IMAGE_H, IMAGE_W, norm)
        nn = Fcn(data, labels, (IMAGE_H, IMAGE_W), CHECKPOINT_PATH, LOG_PATH)
        nn.define_loss()
        nn.load_model(join(model_path, model_name))
        nn.train()
        nn.save_checkpoint()
        return nn

    # def test_model(self, model_name, norm=False):
    #     nn = Fcn(None, None, (IMAGE_H, IMAGE_W), None, None)
    #     nn.load_model(join(model_path, model_name))
    #     test_data, test_names = dataset.get_test_data(TESTING_DATA_DIR, IMAGE_H, IMAGE_W, norm)
    #     result = nn.predict(test_data) # raw matrix of one-hot vectors
    #     # translate into RGB images
    #     rgb_img_list = []
    #     for raw_img in result:
    #         rgb_img = util.transfer_to_rgb(raw_img)
    #         rgb_img_list.append(rgb_img)
    #     util.put_images_to(TESTING_PRED_DIR, rgb_img_list, test_names)
