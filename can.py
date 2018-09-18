# python 2.7
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.misc
import random
import os.path
from os.path import join, expanduser
from glob import glob

import tensorflow as tf
from tensorflow import keras


def gen_batch_function(data_folder, image_h, image_w):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_h, image_w: Tuple - Shape of image
    :return:
    """
    background_color = np.array([255, 255, 255]) # white
    left_lane_color = np.array([255, 0, 0])      # red
    right_lane_color = np.array([0, 0, 255])     # blue

    def get_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'gt_image_2', '*.png'))

        # make sure the label and image are matched
        image_paths.sort()
        label_paths.sort()

        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file_id in range(batch_i, batch_i + batch_size):
                image_file = image_paths[image_file_id]
                gt_image_file = label_paths[image_file_id]

                image = scipy.misc.imread(image_file)
                gt_image = scipy.misc.imread(gt_image_file)

                print(image.shape)
                print(gt_image.shape)
                h, w = image_h, image_w

                gt_bg = np.all(gt_image == background_color, axis=2).reshape(h, w, 1)
                gt_ll = np.all(gt_image == left_lane_color, axis=2).reshape(h, w, 1)
                gt_rl = np.all(gt_image == right_lane_color, axis=2).reshape(h, w, 1)
                gt_image = np.concatenate((gt_bg, gt_ll, gt_rl), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def layers(num_classes):
    """
    Create the layers for a fully convolutional network.
    For reference: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    :return: The Tensor for the last layer of output
    """
    pass


def optimize(net_prediction, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param net_prediction: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Unroll
    logits_flat = tf.reshape(net_prediction, (-1, num_classes))
    labels_flat = tf.reshape(labels, (-1, num_classes))

    # Define loss: Cross Entropy
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat))

    # Define optimization step
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits_flat, train_step, cross_entropy_loss


def train_nn(sess, training_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             image_input, labels, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param training_epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    pass


def run():
    sess.run(tf.global_variables_initializer())
    lr = 1e-4 # learning rate
    examples_each_epoch = 100
    for e in range(0, training_epochs):
        loss_this_epoch = 0.0
        for i in range(0, examples_each_epoch):
            # Load a batch of examples
            batch_x, batch_y = next(get_batches_fn(batch_size))
            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={image_input: batch_x, labels: batch_y, keep_prob: 0.25, learning_rate: lr})
            loss_this_epoch += cur_loss
        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(e, loss_this_epoch / examples_each_epoch))


def test_code():
    image_h, image_w = (48, 160)
    data_dir = join('.', 'data')
    training_data_dir = join(data_dir, 'training')

    # test `gen_batch_function`
    get_batches_fn = gen_batch_function(training_data_dir, image_h, image_w)
    batch_x, batch_y = next(get_batches_fn(1))
    print(batch_y.shape)


if __name__ == '__main__':
    test_code()
