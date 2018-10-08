import io
import socket
import struct
import time
import picamera
import threading
import numpy as np
import cv2
from os.path import join

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


def send_img(cs):
    connection = cs.makefile('wb')
    print('ready to send images')
    try:
        output = SplitFrames(connection)
        # with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
        with picamera.PiCamera(resolution=(220, 160), framerate=30) as camera:
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


def recv_data(s):
    """Wait for results from the server.
    """
    while (True):
        try:
            buffer = s.recv(2048)
            if (buffer is not None):
                dosomething(buffer)
                # TODO
        except:
            print('thread: recv data finish')
            break


def dosomething(buffer):
    packaged_parameters = np.frombuffer(buffer, dtype=np.float64).reshape(14)
    image_id = int(packaged_parameters[0])
    w_left = packaged_parameters[1:4]
    w_right = packaged_parameters[4:7]
    w_middle = packaged_parameters[7:10]
    distance_to_center = packaged_parameters[10]
    distance_at_middle = packaged_parameters[11]
    radian = packaged_parameters[12]
    curvature = packaged_parameters[13]
    print('Received: ', str(image_id))


if __name__ == '__main__':
    cs = socket.socket()
    cs.connect(('192.168.20.104', 8888))
    t1 = threading.Thread(target=send_img, args=(cs,))
    t2 = threading.Thread(target=recv_data, args=(cs,))
    t1.start()
    t2.start()
