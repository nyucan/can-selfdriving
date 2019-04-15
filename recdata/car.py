""" Project Entrance.
"""
from __future__ import absolute_import, division, print_function
import sys
import io
import struct
import time
import picamera
import threading
import socket
from math import atan, floor
from os.path import join

import numpy as np
import cv2
# from PIL import Image

from control.controller import Controller
from util.detect import Detector
from util import img_process
from config import configs
import client

IMG_W = configs['data']['image_width']
IMG_H = configs['data']['image_height']
NUM_OF_POINTS = configs['fitting']['num_of_points']

class Car(object):
    """ Offline car-control, with only one thread.
    """
    def __init__(self):
        self.contorller = Controller()
        self.detector = Detector()
        self.pre_img_id = -1
        self.cur_img_id = -1
        self.has_switched = False

    @staticmethod
    def unpackage_paras(packaged_parameters):
        """ Unpackage the parameters.
            @Paras:
                packaged_parameters: np array
            @Returns:
                distance_to_center
        """
        cur_paras = packaged_parameters[0:13]
        w_left, w_right, w_middle = cur_paras[0:3], cur_paras[3:6], cur_paras[6:9]
        distance_to_center = cur_paras[9]
        distance_at_middle = cur_paras[10]
        radian = cur_paras[11]
        curvature = cur_paras[12]
        stop_signal = (np.all(w_left == np.zeros(3)) and np.all(w_right == np.zeros(3)))
        return distance_to_center, distance_at_middle, curvature, stop_signal

    @staticmethod
    def unpackage_paras_from_buffer(buffer):
        """ Unpackage the parameters from buffer.
            @Paras:
                buffer: str
                    The recv buffer.
                    Note that the default recv size should be 112 (np.array(13, dtype=float64))
            @Returns:
                distance_to_tangent
                angle_of_tangent
        """
        num_of_paras = floor(len(buffer) / 128)
        packaged_parameters = np.frombuffer(buffer, dtype=np.float64).reshape(int(16 * num_of_paras))
        if len(packaged_parameters) < 16:
            return -1, 0, 0, False
        cur_paras = packaged_parameters[0:16]
        image_id = int(cur_paras[0])
        w_left, w_right = cur_paras[1:4], cur_paras[4:7]
        distance_to_tangent = cur_paras[14]
        angle_of_tangent = cur_paras[15]
        stop_signal = (np.all(w_left == np.zeros(3)) and np.all(w_right == np.zeros(3)))
        return image_id, distance_to_tangent, angle_of_tangent, stop_signal

    def send_images(self, connection, stream):
        """ Send images. Single thread, will block.
            Helper function for online mode.
        """
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        stream.seek(0)
        connection.write(stream.read())
        stream.seek(0)
        stream.truncate()

    def recv_parameters(self, client_socket):
        """ Receive parameters from the server. Single thread, will block.
            Helper function for online mode.
        """
        buffer = client_socket.recv(1024)
        if (buffer is not None):
            img_id, d2t, aot, stop_signal = Car.unpackage_paras_from_buffer(buffer)
            if img_id <= self.pre_img_id:
                return
            self.cur_img_id = img_id
            if stop_signal:
                self.contorller.finish_control()
            else:
                # self.contorller.make_decision_with_policy(1, d2t, aot)
                self.contorller.make_decision_with_policy(6, d2t, aot)
            self.pre_img_id = img_id

    def offline_main(self, ori_image, debug=True):
        # ------------- preprocessing -------------
        debug_img = img_process.crop_image(ori_image, 0.45, 0.85)
        debug_img = img_process.down_sample(debug_img, (160, 48))
        image = img_process.binarize(debug_img)
        # -----------------------------------------
        paras = self.detector.get_wrapped_all_parameters(image)
        dc, dm, cur, ss = Car.unpackage_paras(paras)
        dis_2_tan, pt = Detector.get_distance_2_tan(paras[6:9])
        radian_at_tan = atan(paras[14])
        if debug: # display the fitting result in real time
            debug_img = img_process.compute_debug_image(debug_img, IMG_W, IMG_H, NUM_OF_POINTS, pt, paras)
            img_process.show_img(debug_img)
        d_arc = img_process.detect_distance(ori_image)
        print(d_arc)
        if d_arc >= 95 and (not self.has_switched):
            # self.contorller.make_decision_with_policy(1, dis_2_tan, radian_at_tan)
            self.contorller.make_decision_with_policy(6, dis_2_tan, radian_at_tan)
        else:
            self.has_switched = True
            # self.contorller.make_decision_with_policy(5, d_arc, dis_2_tan, radian_at_tan)
            self.contorller.make_decision_with_policy(7, d_arc, dis_2_tan, radian_at_tan)

    def run_offline(self, debug=True):
        stream = io.BytesIO()
        self.contorller.start()
        waitting_for_ob = configs['avoid_collision']
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    ori_image = img_process.img_load_from_stream(stream)
                    self.offline_main(ori_image)
                    stream.seek(0)
                    stream.truncate()

    def run_online(self, ip, port):
        pass

    def run_online_single(self, ip, port):
        client_socket = socket.socket()
        client_socket.connect((ip, port))
        connection = client_socket.makefile('wb')
        first_start = True
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.framerate = 30
            time.sleep(1)
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                # start_time = time.time()
                self.send_images(connection, stream)
                self.recv_parameters(client_socket)
                if first_start:
                    self.contorller.start()
                    first_start = False
                # print('processed img ' + str(self.cur_img_id), time.time() - start_time)
        connection.write(struct.pack('<L', 0))
        connection.close()
        client_socket.close()

    def run_as_fast_as_you_can(self):
        self.contorller.start()

    def stop(self):
        self.contorller.finish_control()
        self.contorller.cleanup()


if __name__ == '__main__':
    try:
        car = Car()
        time.sleep(1)
        car.run_as_fast_as_you_can()
    except KeyboardInterrupt:
        car.stop()
