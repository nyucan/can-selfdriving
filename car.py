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
from PIL import Image

from control.controller import Controller
# from control.car_avoid import CarAvoid
from control.processImage import processImage
from util.detect import Detector
from util import img_process
from config import configs
import client

IMG_W = configs['data']['image_width']
IMG_H = configs['data']['image_height']
NUM_OF_POINTS = configs['fitting']['num_of_points']
LOW_LANE_COLOR = np.uint8([[[0,0,0]]])
UPPER_LANE_COLOR = np.uint8([[[0,0,0]]]) + 40
CENTER_W, CENTER_H = int(IMG_W / 2), int(IMG_H / 2)

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
                self.contorller.make_decision(d2t, aot)
            self.pre_img_id = img_id

    def run_offline(self):
        stream = io.BytesIO()
        counter = 1
        first_start = True
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    ori_image = np.array(Image.open(stream))
                    ori_image = img_process.standard_preprocess(ori_image, crop=True, down=True, f=False, binary=False)
                    image = img_process.standard_preprocess(ori_image, crop=False, down=False, f=True, binary=True)
                    paras = self.detector.get_wrapped_all_parameters(image)
                    dc, dm, cur, ss = Car.unpackage_paras(paras)
                    dis_2_tan, pt = Detector.get_distance_2_tan(paras[6:9])
                    # angle_at_tan = paras[14]
                    radian_at_tan = atan(paras[14])
                    # display the fitting result in real time
                    if configs['debug']:
                        debug_img = img_process.compute_debug_image(ori_image, IMG_W, IMG_H, NUM_OF_POINTS, pt, paras)
                        img_process.show_img(debug_img)
                    if first_start:
                        self.contorller.start()
                        startT = time.time()
                        first_start = False
                    # Control the car according to the parameters
                    if ss:
                        ## Stop the car
                        print('------- stop -------')
                        self.contorller.finish_control()
                        self.contorller.make_decision(0,0,startT)
                    else:
                        ## Turn left or turn right
                        print('making desicion with ', dis_2_tan, radian_at_tan)
                        self.contorller.make_decision(dis_2_tan, radian_at_tan)
                    stream.seek(0)
                    stream.truncate()

    def run_online(self, ip, port):
        pass
    #     cs = socket.socket()
    #     cs.connect((ip, port))
    #     send_img_thread = threading.Thread(target=client.send_img, args=(cs, self.contorller,))
    #     recv_data_thread = threading.Thread(target=client.recv_data, args=(cs, self.contorller,))
    #     send_img_thread.start()
    #     recv_data_thread.start()

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
                start_time = time.time()
                self.send_images(connection, stream)
                self.recv_parameters(client_socket)
                if first_start:
                    self.contorller.start()
                    first_start = False
                print('processed img ' + str(self.cur_img_id), time.time() - start_time)
        connection.write(struct.pack('<L', 0))
        connection.close()
        client_socket.close()

    def stop(self):
        self.contorller.finish_control()
        self.contorller.cleanup()
