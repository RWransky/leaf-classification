'''
Leaf Classifier via Deep Highway Networks

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse
import tensorflow.contrib.slim as slim

from model_helpers import *
from data_helpers import *
from network import *

# Setting the training parameters

# Number of possible actions
actions = 99
# How many experience traces to use for each training step.
batch_size = 32
# Number of training steps
num_steps = 101
load_model = False
# The path to save our model to.
path = "./dhn"


def train():
    # load data
    images, labels = load_data()
    # convert to training and validation sets
    x_train, x_valid, train_labels, valid_labels = split_data(images, labels)
    train_dataset = reformat(x_train)
    valid_dataset = reformat(x_valid)

    tf.reset_default_graph()
    mainN = Network()

    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=5)

    # Make list to store losses
    losses = []
    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)

        for step in range(num_steps):
            print('Processing step {}'.format(step))
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_data = (batch_data/255.0)
            batch_data = (batch_data - np.mean(batch_data)) / np.std(batch_data)
            batch_labels = train_labels[offset:(offset + batch_size)]
            _, lossA, yP, LO = sess.run([mainN.update, mainN.loss, mainN.probs, mainN.label_oh],
                feed_dict={mainN.input_layer: batch_data, mainN.label_layer: batch_labels})
            losses.append(lossA)
            if (step % 10 == 0):
                print('Minibatch loss at step %d: %f' % (step, lossA))
                print('Minibatch accuracy: %.1f%%' % accuracy(yP, LO))
                saver.save(sess, path+'/model-'+str(step)+'.cptk')
                print("Saved Model")
        valid_dataset = (valid_dataset/255.0)
        valid_dataset = (valid_dataset - np.mean(valid_dataset)) / np.std(valid_dataset)
        yP, LO = sess.run([mainN.probs, mainN.label_oh],
            feed_dict={mainN.input_layer: valid_dataset, mainN.label_layer: valid_labels})
        print('Validation accuracy: %.1f%%' % accuracy(yP, LO))
        saver.save(sess, path+'/model-'+str(step)+'.cptk')
        print("Saved Model")
        plt.figure(1)
        plt.title('Training Loss')
        plt.plot(range(len(losses)), losses)
        plt.show()


def test():
    tf.reset_default_graph()
    mainN = Network()

    saver = tf.train.Saver(max_to_keep=5)

    # load data
    images, image_id, species = load_test_data()

    test_dataset = reformat(images)

    with tf.Session() as sess:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Loaded!')

        yP = sess.run([mainN.probs], feed_dict={mainN.input_layer: test_dataset})
        np.save('testProbs', yP)
        print('Completed processing {} test images'.format(str(image_id.shape[0])))
        write_results_to_file(species, image_id, yP)


def main():
    parser = argparse.ArgumentParser(description="Train or run leaf classifier")
    parser.add_argument("-m", "--mode", help="Train / Run", required=True)
    args = vars(parser.parse_args())
    if args['mode'] == 'Train':
        train()
    elif args['mode'] == 'Test':
        test()
    else:
        print(':p Invalid Mode.')


if __name__ == "__main__":
    main()
