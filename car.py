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
from control.processImage import processImage
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
                self.contorller.make_decision_with_policy(1, d2t, aot)
            self.pre_img_id = img_id

    def run_offline(self, debug=True):
        stream = io.BytesIO()
        first_start = True
        waitting_for_ob = configs['avoid_collision']
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    # st = time.time()
                    stream.seek(0)
                    ori_image = img_process.img_load_from_stream(stream)
                    # print('load time:', time.time() - st)
                    # down_img = cv2.resize(cropped_img, dsize=None, fx=0.5, fy=0.5)
                    # ------------- preprocessing -------------
                    # st = time.time()
                    # debug_img = img_process.crop_image(ori_image, 0.4, 0.8)
                    debug_img = img_process.crop_image(ori_image, 0.45, 0.85)
                    debug_img = img_process.down_sample(debug_img, (160, 48))
                    image = img_process.binarize(debug_img)
                    # image = img_process.lane_filter(debug_img, np.uint8([[[0,0,0]]]), np.uint8([[[40,40,40]]]))
                    # print('processing time.:', time.time() - st)
                    # -----------------------------------------
                    # st = time.time()
                    paras = self.detector.get_wrapped_all_parameters(image)
                    dc, dm, cur, ss = Car.unpackage_paras(paras)
                    dis_2_tan, pt = Detector.get_distance_2_tan(paras[6:9])
                    radian_at_tan = atan(paras[14])
                    # print('detect time.:', time.time() - st)
                    if waitting_for_ob:
                        ob = img_process.detect_obstacle(ori_image)
                    # display the fitting result in real time
                    # if configs['debug']:
                    #     # ------------- 1. display fitting result on the fly -------------
                    #     # st = time.time()
                    #     debug_img = img_process.compute_debug_image(debug_img, IMG_W, IMG_H, NUM_OF_POINTS, pt, paras)
                    #     img_process.show_img(debug_img)
                    #     # print('compute debug img time:', time.time() - st)
                    #     # ----------------------------------------------------------------
                    if first_start:
                        self.contorller.start()
                        first_start = False
                    # Control the car according to the parameters
                    if waitting_for_ob and ob:
                        ob = False
                        print("attampting to avoid ...")
                        # self.contorller.collision_avoid(time.time())
                        waitting_for_ob = False
                    # elif ss:
                    #     ## Stop the car
                    #     print('------- stop -------')
                    #     self.contorller.finish_control()
                    else:
                        # st = time.time()
                        ## 1. ADP
                        self.contorller.make_decision_with_policy(1, dis_2_tan, radian_at_tan)
                        ## 2. pure pursuit
                        # l_d, sin_alpha = Detector.get_distance_angle_pp(paras[6:9])
                        # self.contorller.make_decision_with_policy(2, l_d, sin_alpha)
                        ## 3. Car following with ADP
                        # estimated_distance = img_process.detect_distance(ori_image)
                        # self.contorller.make_decision_with_policy(3, dis_2_tan, radian_at_tan, estimated_distance)
                        # print('decision time: ', time.time() - st)
                        ## 4. Car followings
                        # self.contorller.make_decision_with_policy(4, estimated_distance)
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
