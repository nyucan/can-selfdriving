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
from os.path import join
import numpy as np
from PIL import Image

from control.controller import Controller
from util.detect import Detector
from util import img_process
from config import configs
import client


class Car(object):
    """ Offline car-control, with only one thread.
    """
    def __init__(self):
        self.contorller = Controller()
        self.detector = Detector()

    @staticmethod
    def preprocess_image(image):
        """ Perform filter operations to pre-process the image.
        """
        img_cropped = img_process.crop_image(image, 0.45, 0.85)
        img_downsampled = img_process.down_sample(img_cropped, (160, 48))
        lane_img = img_process.lane_filter(img_downsampled, LOW_LANE_COLOR, UPPER_LANE_COLOR)
        bin_img = lane_img / 255
        return bin_img, img_downsampled

    @staticmethod
    def unpackage_paras(packaged_parameters):
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

    @staticmethod
    def save_fitting_img(img, wrapped_parameters, img_name):
        debug_img = Detector.mark_image_with_parameters(img, wrapped_parameters)
        img_process.img_save(debug_img, img_name)

    def calc_para_from_image(self, image):
        wrapped_parameters = self.detector.get_wrapped_all_parameters(image)
        return wrapped_parameters

    def make_decisiton_with(self, dc, dm, cur, stop_signal):
        print('making desicion with ', dc, dm, cur ,str(stop_signal))
        if stop_signal:
            # stop the car!
            self.contorller.finish_control()
        else:
            self.contorller.make_decision(dc, dm, cur)

    def run_offline(self):
        stream = io.BytesIO()
        counter = 1
        first_start = True
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    image = np.array(Image.open(stream))
                    processed_image, input_img = Car.preprocess_image(image)
                    paras = self.calc_para_from_image(processed_image)
                    dc, dm, cur, ss = Car.unpackage_paras(paras)
                    ########### visualize result ###########
                    if configs['debug']:
                        w_l, w_r, w_m = paras[0:3], paras[3:6], paras[6:9]
                        x = np.linspace(0, 48, 48)
                        for cur_w in [w_l, w_r, w_m]:
                            pts = np.array([Detector.calc_fitting_pts(cur_w, x), x], np.int32).transpose()
                            input_img = img_process.plot_line(input_img, pts, 'yellow')
                        img_name = join('.', 'test-output', str(counter) + '.png')
                        counter += 1
                        print('save image to ' + img_name)
                        img_process.img_save(input_img, img_name)
                    ########################################
                    if first_start:
                        self.contorller.start()
                        first_start = False
                    self.make_decisiton_with(dc, dm, cur, ss)
                    stream.seek(0)
                    stream.truncate()

    def run_online(self, ip, port):
        # self.contorller.start()
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

    def stop(self):
        self.contorller.finish_control()
        self.contorller.cleanup()


def test_offline():
    try:
        car = Car()
        car.run_offline()
    except KeyboardInterrupt:
        car.stop()


def test_online_multithread(ip, addr):
    try:
        car = Car()
        car.run_online(ip, addr)
    except KeyboardInterrupt:
        car.stop()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please provide running mode')
    else:
        mode = str(sys.argv[1])
        print('running mode: ' + str(mode))
        if mode == 'online':
            test_online_multithread('192.168.20.103', 8888)
        elif mode == 'offline':
            test_offline()
        else:
            print('mode can only be online or offline')
