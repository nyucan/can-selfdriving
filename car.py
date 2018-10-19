""" Offline Version.
"""

import io
import socket
import struct
import time
import picamera
import threading
import numpy as np
import cv2
from os.path import join

from control.controller import Controller
from util.detect import Detector


class Car(object):
    def __init__(self):
        self.contorller = Controller()
        self.detector = Detector()

    @classmethod
    def preprocess_image(cls, image):
        """ Perform filter operations to pre-process the image.
        """
        return image

    @classmethod
    def unpackage_paras(cls, packaged_parameters):
        """ Unpackage the parameters from buffer.
            @Paras:
                packaged_parameters: np array
            @Returns:
                image_id
                distance_to_center
        """
        cur_paras = packaged_parameters[0:14]
        image_id = int(cur_paras[0])
        w_left, w_right, w_middle = cur_paras[1:4], cur_paras[4:7], cur_paras[7:10]
        distance_to_center = cur_paras[10]
        distance_at_middle = cur_paras[11]
        radian = cur_paras[12]
        curvature = cur_paras[13]
        stop_signal = (np.all(w_left == np.zeros(3)) and np.all(w_right == np.zeros(3)))
        return image_id, distance_to_center, distance_at_middle, curvature, stop_signal

    def calc_para_from_image(self, image):
        wrapped_parameters = self.detector.get_wrapped_all_parameters(image)
        debug_img = Detector.mark_image_with_parameters(predicted_img, wrapped_parameters)
        ## debug
        cv2.imwrite('./test-output/' + str(imageId) + '.png', debug_img)
        return wrapped_parameters

    def make_decisiton_with(self, dc, dm, cur, stop_signal):
        print('making desicion with ', dc, dm, cur ,str(stop_signal))
        if stop_signal:
            # stop the car!
            self.contorller.finish_control()
        else:
            self.contorller.make_decision(dc, dm, cur)

    def run(self):
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                self.contorller.motor.motor_startup()
                for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    processed_image = Car.preprocess_image(frame)
                    paras = Car.calc_para_from_image(processed_image)
                    image_id, dc, dm, cur, ss = Car.unpackage_paras(paras)
                    print('processed image: ' + str(image_id))
                    self.make_decisiton_with(dc, dm, cur, ss)


if __name__ == '__main__':
    car = Car()
    car.run()

