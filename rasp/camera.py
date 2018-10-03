# python 2.7
from __future__ import absolute_import, division, print_function

from picamera import PiCamera
from time import sleep


class CameraController(object):
    def __init__(self):
        self.camera = PiCamera()


    def capture(self):
        # self.camera.start_preview()
        # sleep(1)
        # self.camera.capture('/home/pi/Desktop/image.jpg')
        # self.camera.stop_preview()
        pass
