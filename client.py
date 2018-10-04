import io
import socket
import struct
import time
import picamera
import threading
import pickle
import numpy
import cv2
from util import detect

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
    try:
        output = SplitFrames(connection)
        with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
            time.sleep(2)
            start = time.time()
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(30)
            camera.stop_recording()
            # Write the terminating 0-length to the connection to let the server know we're done
            connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        client_socket.close()
        finish = time.time()
        print('Sent %d images in %d seconds at %.2ffps' % (output.count, finish-start, output.count / (finish-start)))


def recv_data(s):
    # wait for result from the server
    while (True):
        buffer = s.recv(4096)
        dosomething(buffer)


def dosomething(buffer):
    # placeholder
    data = pickle.loads(buffer)
    image_id = data[0]
    pts_left, pts_right = data[1][0], data[1][1]
    blank_image = np.zeros((48, 160, 3), np.uint8)
    fitted_img = detect.plot_lines(blank_image, pts_left, pts_right)
    cv2.imwrite(join('.', 'comm', str(image_id) + '.png'), fitted_img)
    # print('Received', repr(data))


if __name__ == '__main__':
    cs = socket.socket()
    cs.connect(('192.168.20.113', 8000))
    t1 = threading.Thread(target=send_img, args=(cs,))
    t2 = threading.Thread(target=recv_data, args=(cs,))
    t1.start()
    t2.start()
