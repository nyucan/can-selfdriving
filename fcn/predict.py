# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import join
from glob import glob
import numpy as np

from fcn.fcn import Fcn

MODEL_PATH = join('.', 'models')
IMAGE_H, IMAGE_W = 48, 160

class Predictor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if (model_name is not None):
            self.model = Fcn(None, None, (IMAGE_H, IMAGE_W), None, None)
            self.model.load_model(join(MODEL_PATH, model_name))
        else:
            self.model = None

    def predict(self, image, norm=False):
        predict_input = np.array([image])
        raw_result = self.model.predict(predict_input)[0]
        result = self.translate_res(raw_result)
        return result

    def translate_res(self, raw_res):
        h, w, c = raw_res.shape
        res_img = np.uint8(np.ones((h, w, 3))) * 255 # default: white
        for i in range(h):
            for j in range(w):
                if np.all(raw_res[i][j] == [0, 1]):
                    res_img[i][j] = [0, 0, 0]
        return res_img
