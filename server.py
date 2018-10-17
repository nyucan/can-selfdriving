# python 2.7
import io
import socket
import struct
import numpy as np
import cv2
from time import sleep
from PIL import Image

from fcn.predict import Predictor
from util.detect import Detector


class Server(object):
    def __init__(self, model):
        self.predictor = Predictor(model)
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
            print('server: receive None')
            return None
        # Construct a stream to hold the image data and read the image data from the connection
        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream).convert('RGB')
        open_cv_image = np.array(image)
        cv2.imwrite('./comm/' + 'ori.png', open_cv_image)
        open_cv_image = Server.crop_image(open_cv_image, 0.35, 0.75)
        # open_cv_image = Server.crop_image(open_cv_image, 0.37, 0.77)
        open_cv_image = cv2.resize(open_cv_image, (160, 48), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('./comm/' + 'ori-2.png', open_cv_image)
        return open_cv_image

    def listen(self):
        print('server: listening ...')
        try:
            image_id = 0
            while True:
                new_img = self.recv_images()
                if (new_img is None):
                    break
                print('Server: transmited image ' + str(image_id))
                packaged_parameters = self.predict_and_fit(image_id, new_img)
                print('Server: predicted and fitted image ' + str(image_id))
                packaged_parameters_with_id = np.concatenate(([image_id], packaged_parameters))
                s_packaged_parameters = packaged_parameters_with_id.tobytes()
                self.s.sendall(s_packaged_parameters)
                image_id = image_id + 1
        except:
            print('closed by thread')
        finally:
            self.close_connection()
            print('connection closed')

    def predict_and_fit(self, imageId, image):
        """ Make prediction and then fit the predicted image.
            @return: image, left_parameters, left_parameters
        """
        cv2.imwrite('./comm/1/' + str(imageId) + '.png', image)
        # predict
        predicted_img = self.predictor.predict(image)
        cv2.imwrite('./comm/2/' + str(imageId) + '.png', predicted_img)

        # fit
        wrapped_parameters = self.detector.get_wrapped_all_parameters(predicted_img)
        debug_img = Detector.mark_image_with_parameters(predicted_img, wrapped_parameters)
        cv2.imwrite('./comm/3/' + str(imageId) + '.png', debug_img)
        return wrapped_parameters

    def close_connection(self):
        self.connection.close()
        self.server.close()

    @classmethod
    def crop_image(cls, img, lower_bound, upper_bound):
        """ Crop an image in the vertical direction with lower and upper bound.
        """
        img_cropped = img[int(img.shape[1]*lower_bound):int(img.shape[1]*upper_bound),:]
        return img_cropped


if __name__ == '__main__':
    s = Server('1538680331.7627041')
    s.listen()

