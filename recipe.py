# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import join, expanduser
from glob import glob
import time
import numpy as np
import cv2
import pickle

from fcn.fcn import Fcn
from util import detect
from util import dataset
from util import util

# path
image_h, image_w = (48, 160)
training_data_dir = join('.', 'data', 'training')
data_folder = join(training_data_dir, 'aug-data')
label_folder = join(training_data_dir, 'aug-label')
testing_data_dir = join('.', 'data', 'testing', 'image')
testing_pred_dir = join('.', 'data', 'testing', 'predict')

checkpoint_path = join('.', 'models')
log_path = join('.', 'logs')
model_path = join('.', 'models')

class Predictior(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if (model_name is not None):
            self.model = Fcn(None, None, (image_h, image_w), None, None)
            self.model.load_model(join(model_path, model_name))
        else:
            self.model = None

    def predict(self, image, norm=False):
        predict_input = np.array([image])
        if (norm):
            normalized_input = dataset.normalize(predict_input)
            raw_result = self.model.predict(normalized_input)[0]
        else:
            raw_result = self.model.predict(predict_input)[0]
        result = util.transfer_to_rgb(raw_result)
        return result


def train_nn_from_sketch(norm=False):
    data, labels = dataset.get_data(data_folder, label_folder, image_h, image_w, norm)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.build_layers()
    nn.train()
    nn.save_checkpoint()
    return nn


def train_nn_from_model(model_name, norm=False):
    data, labels = dataset.get_data(data_folder, label_folder, image_h, image_w, norm)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.load_model(join(model_path, model_name))
    nn.train()
    nn.save_checkpoint()
    return nn


def test_model(model_name, norm=False):
    nn = Fcn(None, None, (image_h, image_w), None, None)
    nn.load_model(join(model_path, model_name))
    test_data, test_names = dataset.get_test_data(testing_data_dir, image_h, image_w, norm)
    result = nn.predict(test_data) # raw matrix of one-hot vectors
    # translate into RGB images
    rgb_img_list = []
    for raw_img in result:
        rgb_img = util.transfer_to_rgb(raw_img)
        rgb_img_list.append(rgb_img)
    util.put_images_to(testing_pred_dir, rgb_img_list, test_names)


def fit(image):
    """  No usage any more!
    """
    image, pts_left, pts_right = detect.fit_image(image)
    return pts_left, pts_right


def get_fitting_parameters(image):
    """ No usage any more!
        Wrap the parameters.
        @return wrapped_parameters: np.array
            [0:3]  :  w_l
            [3:6]  :  w_r
            [6:9]  :  w_m
            [9]    :  dc
            [10]   :  dm
            [11]   :  radian
            [12]   :  curvature
    """
    w_l, w_r, w_m, dc, dm, r, c = detect.get_client_parameters(image)
    wrapped_parameters = np.array([], dtype=np.float64)
    wrapped_parameters = np.concatenate((wrapped_parameters, w_l, w_r, w_m, [dc, dm, r, c]))
    return wrapped_parameters


def _test():
    p = Predictior('1538586986.23')
    test_img = util.get_an_image_from(join('.', 'data', 'aug', 'output', '342.png'))
    predicted_img = p.predict(test_img)
    pts_left, pts_right = fit(predicted_img)
    blank_image = np.zeros((48, 160, 3), np.uint8)
    fitted_img = detect.plot_lines(blank_image, pts_left, pts_right)
    cv2.imwrite(join('.', 'comm', str(1) + '.png'), fitted_img)


if __name__ == '__main__':
    # nn = train_nn_from_sketch()
    # nn = train_nn_from_model('1538672160.36')
    test_model('1538680331.7627041')
