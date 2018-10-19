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
from PIL import Image
from os.path import join

from control.controller import Controller
from util.detect import Detector


LOW_LANE_COLOR = np.uint8([[[0,0,0]]])
UPPER_LANE_COLOR = np.uint8([[[0,0,0]]]) + 40
cur_img = None

class Car(object):
    def __init__(self):
        self.contorller = Controller()
        self.detector = Detector()
        self.counter = 1

    @classmethod
    def lane_filter(cls, img, lower_lane_color, upper_lane_color):
        """ Use color filter to show lanes in the image.
        """
        laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
        return laneIMG

    @classmethod
    def crop_image(cls, img, lower_bound, upper_bound):
        img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
        return img_cropped

    @classmethod
    def preprocess_image(cls, image):
        """ Perform filter operations to pre-process the image.
        """
        img_cropped = Car.crop_image(image, 0.45, 0.85)
        img_downsampled = cv2.resize(img_cropped, (160, 48), interpolation=cv2.INTER_LINEAR)
        ## for debug only
        cur_img = img_downsampled
        ######################
        lane_img = Car.lane_filter(img_downsampled, LOW_LANE_COLOR, UPPER_LANE_COLOR)
        bin_img = lane_img / 255
        return bin_img

    @classmethod
    def unpackage_paras(cls, packaged_parameters):
        """ Unpackage the parameters from buffer.
            @Paras:
                packaged_parameters: np array
            @Returns:
                distance_to_center
        """
        # print(packaged_parameters)
        cur_paras = packaged_parameters[0:13]
        w_left, w_right, w_middle = cur_paras[0:3], cur_paras[3:6], cur_paras[6:9]
        distance_to_center = cur_paras[9]
        distance_at_middle = cur_paras[10]
        radian = cur_paras[11]
        curvature = cur_paras[12]
        stop_signal = (np.all(w_left == np.zeros(3)) and np.all(w_right == np.zeros(3)))
        return distance_to_center, distance_at_middle, curvature, stop_signal

    @classmethod
    def save_fitting_img(cls, img, wrapped_parameters, img_name):
        debug_img = Detector.mark_image_with_parameters(img, wrapped_parameters)
        cv2.imwrite(img_name, debug_img)

    def calc_para_from_image(self, image):
        wrapped_parameters = self.detector.get_wrapped_all_parameters(image)
        img_name = './test-output/' + str(self.counter) + '.png'
        Car.save_fitting_img(cur_img, wrapped_parameters, img_name)
        self.counter += 1
        return wrapped_parameters

    def make_decisiton_with(self, dc, dm, cur, stop_signal):
        print('making desicion with ', dc, dm, cur ,str(stop_signal))
        if stop_signal:
            # stop the car!
            self.contorller.finish_control()
        else:
            self.contorller.make_decision(dc, dm, cur)

    def run(self):
        stream = io.BytesIO()
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                self.contorller.motor.motor_startup()
                for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    image = np.array(Image.open(stream))
                    processed_image = Car.preprocess_image(image)
                    paras = self.calc_para_from_image(processed_image)
                    dc, dm, cur, ss = Car.unpackage_paras(paras)
                    self.make_decisiton_with(dc, dm, cur, ss)
                    stream.seek(0)
                    stream.truncate()

    def stop(self):
        self.contorller.finish_control()
        self.contorller.cleanup()


if __name__ == '__main__':
    car = Car()
    car.run()
    try:
        car = Car()
        car.run()
    except KeyboardInterrupt:
        car.stop()
