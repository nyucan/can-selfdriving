# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import join, expanduser
from glob import glob
import time
import numpy as np
import cv2

from fcn.fcn import Fcn
from fcn import dataset

IMAGE_H, IMAGE_W = (48, 160)
DATA_DIR = join('.', 'data', 'training', 'aug-data')
LABEL_DIR = join('.', 'data', 'training', 'aug-label')
TESTING_DATA_DIR = join('.', 'data', 'testing', 'image')
TESTING_PRED_DIR = join('.', 'data', 'testing', 'predict')
CHECKPOINT_PATH = join('.', 'models')
LOG_PATH = join('.', 'logs')

class Trainer(object):
    def __init__(self):
        pass

    @classmethod
    def train_nn_from_sketch(cls, norm=False):
        data, labels = dataset.get_data(DATA_DIR, LABEL_DIR, IMAGE_H, IMAGE_W, norm)
        nn = Fcn(data, labels, (IMAGE_H, IMAGE_W), CHECKPOINT_PATH, LOG_PATH)
        nn.define_loss()
        nn.build_layers()
        nn.train()
        nn.save_checkpoint()
        return nn

    @classmethod
    def train_nn_from_model(self, model_name, norm=False):
        data, labels = dataset.get_data(DATA_DIR, LABEL_DIR, IMAGE_H, IMAGE_W, norm)
        nn = Fcn(data, labels, (IMAGE_H, IMAGE_W), CHECKPOINT_PATH, LOG_PATH)
        nn.define_loss()
        nn.load_model(join(model_path, model_name))
        nn.train()
        nn.save_checkpoint()
        return nn
