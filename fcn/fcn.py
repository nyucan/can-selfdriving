# python 2.7
# the fcn model for semantic segmentation
from __future__ import absolute_import, division, print_function
import os.path
from os.path import join
import time

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras


class Fcn(object):
    # constructor
    def __init__(self, data, labels, img_size, checkpoint_path, log_path):
        # hyperparameters
        self.image_h, self.image_w = img_size
        self.num_classes = 2
        self.training_epochs = 5
        self.batch_size = 10
        self.learning_rate = 0.00001
        self.pos_weight = 5
        self.loss_func = None
        self.model = None

        # dataset
        self.data = data
        self.labels = labels

        # path
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path


    def build_layers(self):
        nc = self.num_classes
        inputs = keras.Input(shape=(48, 160, 3))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inputs)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv1)
        conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
        conv4 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv3)

        # skip connection: merge conv1 with conv4
        conv1_out = keras.layers.Conv2D(filters=nc, kernel_size=(3, 3), padding='same')(conv1)
        conv4_out = keras.layers.Conv2D(filters=nc, kernel_size=(3, 3), padding='same')(conv4)
        merge1 = keras.layers.Add()([conv1_out, conv4_out])
        conv5 = keras.layers.Conv2D(filters=nc, kernel_size=(1, 1), padding='same')(merge1)

        predictions = keras.layers.Softmax(axis=3)(conv5)
        self.model = keras.Model(inputs=inputs, outputs=predictions)


    def define_loss(self):
        pw = self.pos_weight
        nc = self.num_classes
        def loss(y_true, y_pred):
            # flatten data
            logits_flat = tf.reshape(y_pred, (-1, nc))
            labels_flat = tf.reshape(y_true, (-1, nc))

            # Define loss
            # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_flat, logits=logits_flat))
            cross_entropy_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels_flat, logits=logits_flat, pos_weight=pw))
            return cross_entropy_loss
        self.loss_func = loss


    def train(self):
        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(log_dir=self.log_path)
        ]
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate), loss=self.loss_func, metrics=['mae', 'acc'])
        self.model.fit(self.data, self.labels, epochs=self.training_epochs, callbacks=callbacks, batch_size=self.batch_size)


    def predict(self, images):
        result = self.model.predict(images, batch_size=1)
        return result


    def save_checkpoint(self):
        t = time.time()
        keras.models.save_model(self.model, join(self.checkpoint_path, str(t)), overwrite=True, include_optimizer=True)


    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path, custom_objects=None, compile=True)
