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
from math import atan
from os.path import join
import numpy as np
from PIL import Image

from control.controller import Controller
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

    @staticmethod
    def unpackage_paras(packaged_parameters):
        """ Unpackage the parameters from buffer.
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

    def run_offline(self):
        stream = io.BytesIO()
        counter = 1
        first_start = True
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    ori_image = np.array(Image.open(stream))
                    image = img_process.standard_preprocess(ori_image)
                    paras = self.detector.get_wrapped_all_parameters(image)
                    dc, dm, cur, ss = Car.unpackage_paras(paras)
                    dis_2_tan, pt = Detector.get_distance_2_tan(paras[6:9])
                    angle_at_tan = Detector.get_angle_of_tan(paras[6:9], pt)
                    ### calculate points
                    w = paras[6:9]
                    delta_y = 15
                    delta_x = angle_at_tan * delta_y
                    from_pt = (int(pt[0] - delta_x), pt[1] - delta_y)
                    to_pt = (int(pt[0] + delta_x), pt[1] + delta_y)
                    ################# visualize result #################
                    # display the fitting result in real time
                    if configs['debug']:
                        debug_img = img_process.crop_image(ori_image, configs['data']['crop_part'][0], configs['data']['crop_part'][1])
                        debug_img = img_process.down_sample(debug_img, (IMG_W, IMG_H))
                        debug_img = img_process.mark_image_with_parameters(debug_img, paras, IMG_H, NUM_OF_POINTS)
                        img_process.mark_image_with_pt(debug_img, (CENTER_W, CENTER_H), (0, 255, 0))
                        img_process.mark_image_with_pt(debug_img, pt, (0, 255, 255))
                        print(dis_2_tan, atan(angle_at_tan))
                        img_process.mark_image_with_line(debug_img, from_pt, to_pt)
                        debug_img = img_process.enlarge_img(debug_img, 6)
                        img_process.show_img(debug_img)
                    ####################################################
                    if first_start:
                        self.contorller.start()
                        first_start = False
                    # Control the car according to the parameters
                    if ss:
                        ## Stop the car
                        print('------- stop -------')
                        self.contorller.finish_control()
                    else:
                        ## Turn left or turn right
                        print('making desicion with ', dc, dm, cur)
                        self.contorller.make_decision(dc, dm, dis_2_tan, cur)
                    stream.seek(0)
                    stream.truncate()

    def run_online(self, ip, port):
        cs = socket.socket()
        cs.connect((ip, port))
        send_img_thread = threading.Thread(target=client.send_img, args=(cs, self.contorller,))
        recv_data_thread = threading.Thread(target=client.recv_data, args=(cs, self.contorller,))
        send_img_thread.start()
        recv_data_thread.start()

    def run_online_alone(self, ip, port):
        client_socket = socket.socket()
        client_socket.connect((ip, port))
        connection = client_socket.makefile('wb')
        try:
            with picamera.PiCamera() as camera:
                camera.resolution = (640, 480)
                camera.framerate = 30
                time.sleep(2)
                start = time.time()
                count = 0
                stream = io.BytesIO()
                for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                    connection.write(struct.pack('<L', stream.tell()))
                    connection.flush()
                    stream.seek(0)
                    connection.write(stream.read())
                    count += 1
                    if time.time() - start > 30:
                        break
                    stream.seek(0)
                    stream.truncate()
            connection.write(struct.pack('<L', 0))
        finally:
            connection.close()
            client_socket.close()
            finish = time.time()
            print(finish - start)

    def stop(self):
        self.contorller.finish_control()
        self.contorller.cleanup()
