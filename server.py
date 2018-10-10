# python 2.7
import io
import socket
import struct
# import pickle
import numpy as np
import cv2
from PIL import Image

import recipe
from util.detect import Detector


class Server(object):
    def __init__(self, model):
        self.predictor = recipe.Predictior(model)
        self.detector = Detector()
        self.server = socket.socket()
        self.server.bind(('0.0.0.0', 8888))
        self.server.listen(0)
        print('server: waitting for connection')

        self.s = self.server.accept()[0]
        self.connection = self.s.makefile('rb')
        print('server: new connection')

    def recv_images(self):
        """ Get image from the server.
            @returns
                return `None` if no more images
        """
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            return None
        # Construct a stream to hold the image data and read the image data from the connection
        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream).convert('RGB')
        open_cv_image = np.array(image)
        open_cv_image = cv2.resize(open_cv_image, (220, 160))
        open_cv_image = open_cv_image[60:108, 30:190]
        return open_cv_image

    def listen(self):
        print('server: listening ...')
        try:
            image_id = 0
            while True:
                new_img = self.recv_images()
                if (new_img is None):
                    break
                cv2.imwrite('./comm/' + str(image_id) + '.png', new_img)
                print('transmited image ' + str(image_id))
                packaged_parameters = self.predict_and_fit(new_img)
                packaged_parameters_with_id = np.concatenate(([image_id], packaged_parameters))
                s_packaged_parameters = packaged_parameters_with_id.tobytes()
                self.s.sendall(s_packaged_parameters)
                image_id = image_id + 1
        except:
            print('closed by thread')
        finally:
            self.close_connection()
            print('connection closed')

    def predict_and_fit(self, image):
        """ Make prediction and then fit the predicted image.
            @return: image, left_parameters, left_parameters
        """
        predicted_img = self.predictor.predict(image)
        wrapped_parameters = self.detector.get_wrapped_all_parameters(predicted_img)
        # wrapped_parameters = recipe.get_fitting_parameters(predicted_img)
        return wrapped_parameters

    def close_connection(self):
        self.connection.close()
        self.server.close()


if __name__ == '__main__':
    s = Server('1538680331.7627041')
    s.listen()

