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


class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)
        self.stream.write(buf)


class Client(object):
    def __init__(self):
        pass


def send_img(cs):
    connection = cs.makefile('wb')
    print('client: sending images')
    try:
        output = SplitFrames(connection)
        # with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
        with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
            time.sleep(2)
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(10)
            camera.stop_recording()
            # Write the terminating 0-length to the connection to let the server know we're done
            connection.write(struct.pack('<L', 0))
    finally:
        time.sleep(2)
        connection.close()
        cs.close()
        print('connection closed')


def recv_data(s, contorller):
    """Wait for results from the server.
    """
    print('client: ready to recv data')
    pre_img_id = -1
    contorller.motor.motor_startup()
    while (True):
        # try:
        buffer = s.recv(2048)
        if (buffer is not None):
            img_id, dc, dm, cur, signal = unpackage_paras(buffer)
            print('Received: ' + str(img_id))
            if img_id <= pre_img_id:
                # outdate data
                continue
            make_decisiton_with(dc, dm, cur, signal, contorller)
            pre_img_id = img_id  # update
        # except:
        #     print('thread: recv data finish')
        #     break


def unpackage_paras(buffer):
    """ Unpackage the parameters from buffer.
        @Returns:
            image_id
            distance_to_center
    """
    packaged_parameters = np.frombuffer(buffer, dtype=np.float64).reshape(14)
    image_id = int(packaged_parameters[0])
    w_left = packaged_parameters[1:4]
    w_right = packaged_parameters[4:7]
    w_middle = packaged_parameters[7:10]
    distance_to_center = packaged_parameters[10]
    distance_at_middle = packaged_parameters[11]
    radian = packaged_parameters[12]
    curvature = packaged_parameters[13]

    stop_signal = (np.all(w_left == np.zeros(3)) and np.all(w_right == np.zeros(3)))
    return image_id, distance_to_center, distance_at_middle, curvature, stop_signal


def make_decisiton_with(dc, dm, cur, stop_signal, contorller):
    print('making desicion with ', dc, dm, cur ,str(stop_signal))
    if stop_signal:
        # stop the car!
        contorller.finish_control()
    else:
        contorller.make_decision(dc, dm, cur)
        # both sides were detected


def main():
    # to store the previous curves for left and right lane
    w_left_previous, w_right_previous = np.zeros((3)), np.zeros((3))

    # setup socket connection
    cs = socket.socket()
    cs.connect(('192.168.20.104', 8888))

    # setup controller
    contorller = Controller()

    # try:
    # create threads
    send_img_thread = threading.Thread(target=send_img, args=(cs,))
    recv_data_thread = threading.Thread(target=recv_data, args=(cs, contorller,))

    # start all threads
    send_img_thread.start()
    recv_data_thread.start()
    # except:
    #     contorller.motor.motor_stop()


if __name__ == '__main__':
    main()
