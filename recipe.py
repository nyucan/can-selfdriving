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


image_h, image_w = (48, 160)
training_data_dir = join('.', 'data', 'training')
data_folder = join(training_data_dir, 'aug-data')
label_folder = join(training_data_dir, 'aug-label')
testing_data_dir = join('.', 'data', 'testing', 'image')
testing_pred_dir = join('.', 'data', 'testing', 'predict')

checkpoint_path = join('.', 'models')
log_path = join('.', 'logs')
model_path = join('.', 'models')


def train_nn_from_sketch():
    data, labels = dataset.get_data(data_folder, label_folder, image_h, image_w)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.build_layers()
    nn.train()
    nn.save_checkpoint()
    return nn


def train_nn_from_model(model_name):
    data, labels = dataset.get_data(data_folder, label_folder, image_h, image_w)
    nn = Fcn(data, labels, (image_h, image_w), checkpoint_path, log_path)
    nn.define_loss()
    nn.load_model(join(model_path, model_name))
    nn.train()
    nn.save_checkpoint()
    return nn


def test_model(model_name):
    nn = Fcn(None, None, (image_h, image_w), None, None)
    nn.load_model(join(model_path, model_name))
    test_data, test_names = dataset.get_test_data(testing_data_dir, image_h, image_w)
    result = nn.predict(test_data) # raw matrix of one-hot vectors
    # translate into RGB images
    rgb_img_list = []
    for raw_img in result:
        rgb_img = util.transfer_to_rgb(raw_img)
        rgb_img_list.append(rgb_img)
    util.put_images_to(testing_pred_dir, rgb_img_list, test_names)


# predict the lane of an image
# def make_prediction_with(model, target_image, output_dir):
#     nn = Fcn(None, None, (image_h, image_w), None, None)
#     nn.load_model(join(model_path, model_name))


def predict(model_name, image):
    predict_input = np.array([image])
    nn = Fcn(None, None, (image_h, image_w), None, None)
    nn.load_model(join(model_path, model_name))
    raw_result = nn.predict(predict_input)[0]
    result = util.transfer_to_rgb(raw_result)
    return result


def fit(image):
    image, pts_left, pts_right = detect.fit_image(image)
    return pts_left, pts_right


if __name__ == '__main__':
    # nn = train_nn_from_sketch()
    # nn = train_nn_from_model('1538586986.23')
    # test_model('1538068266.3')
    test_img = util.get_an_image_from(join('.', 'data', 'aug', 'output', '342.png'))
    predicted_img = predict('1538586986.23', test_img)
    pts_left, pts_right = fit(predicted_img)
    result = np.array([pts_left, pts_right])
    result_with_id = [1, result]
    print(len(pickle.dumps(result_with_id)))
