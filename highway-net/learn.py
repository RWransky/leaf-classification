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
    # We define LSTM cells for primary reinforcement networks
    cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
    mainN = Network(h_size, cell, 'main')

    saver = tf.train.Saver(max_to_keep=5)

    # initialize test state
    test = TestState()

    with tf.Session() as sess:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        probList = []

        for i in range(1):
            # Reset environment and get first new observation
            sP, num_tests, image_id, species = test.reset()
            s = processState(sP)
            j = 0
            # Reset the recurrent layer's hidden state
            state = (np.zeros([1, h_size]), np.zeros([1, h_size]))

            # The Deep Reinforcement Network
            while j < num_tests:
                j += 1
                a, state1 = sess.run([mainN.softMaxAdv, mainN.rnn_state],
                    feed_dict={mainN.scalarInput: [s/255.0],
                    mainN.trainLength: 1, mainN.state_in: state, mainN.batch_size: 1})
                # a = a/np.max(a)
                probList.append('{},'.format(str(image_id)) + convert_list_of_ints_to_string(a.tolist()))

                if j < num_tests:
                    s1P, image_id = test.frame_step(a)
                    s1 = processState(s1P)

                    s = s1
                    sP = s1P
                    state = state1

            print('Completed processing {} test images'.format(str(num_tests)))
            write_results_to_file(str(i), species, probList)


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
