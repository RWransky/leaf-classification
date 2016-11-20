'''
Leaf Classifier via Deep Reinforcement Learning

Initial code inspired by @awjuliani

Extensive changes to game model and training methods

'''

import tensorflow as tf
import os
import argparse
import tensorflow.contrib.slim as slim

from model_helpers import *
from episode_recorder import *
from network import *
from game import *

# Setting the training parameters

# Number of possible actions
actions = 99
# How many experience traces to use for each training step.
batch_size = 4
# How long each experience trace will be when training
trace_length = 8
# How often to perform a training step.
update_freq = 5
# Discount factor on the target Q-values
y = .99
# Starting chance of random action
startE = 1
# Final chance of random action
endE = 0.1
# How many steps of training to reduce startE to endE.
anneling_steps = 10000
# How many episodes of game environment to train network with.
num_episodes = 10000
# How many steps of random actions before training begins.
pre_train_steps = 10000
# Whether to load a saved model.
load_model = False
# The path to save our model to.
path = "./drqn"
# The size of the final convolutional layer before splitting it into Advantage and Value streams.
h_size = 512
# The max allowed length of our episode.
max_epLength = 100
# Number of epidoes to periodically save for analysis
summaryLength = 100


def train():
    tf.reset_default_graph()
    # We define LSTM cells for primary and target reinforcement networks
    cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
    mainN = Network(h_size, cell, 'main')
    targetN = Network(h_size, cellT, 'target')

    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=5)

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables)

    myRecorder = episode_recorder()

    # Determine the rate of decreasing random actions
    e = startE
    stepDrop = (startE - endE)/anneling_steps

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)

        # Set the target network to be equal to the primary network.
        updateTarget(targetOps, sess)

        for i in range(num_episodes):
            episodeBuffer = []
            # Reset environment and get first new observation
            sP = game.GameState.reset()
            s = processState(sP)
            d = False
            rAll = 0
            j = 0
            # Reset the recurrent layer's hidden state
            state = (np.zeros([1, h_size]), np.zeros([1, h_size]))

            # The Deep Reinforcement Network
            while j < max_epLength:
                j += 1
                # Choose an action either randomly or from prediction
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    state1 = sess.run(mainN.rnn_state,
                        feed_dict={mainN.scalarInput: [s/255.0],
                        mainN.trainLength: 1, mainN.state_in: state, mainN.batch_size: 1})
                    a = np.random.randint(0, actions)
                else:
                    a, state1 = sess.run([mainN.predict, mainN.rnn_state],
                        feed_dict={mainN.scalarInput: [s/255.0],
                        mainN.trainLength: 1, mainN.state_in: state, mainN.batch_size: 1})
                    a = a[0]
                s1P, r, d = game.GameState.frame_step(a)
                s1 = processState(s1P)
                total_steps += 1
                episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % (update_freq*1000) == 0:
                        print("Target network updated.")
                        updateTarget(targetOps, sess)

                    if total_steps % (update_freq) == 0:
                        # Reset the recurrent layer's hidden state
                        state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
                        # Get a random batch of experiences.
                        trainBatch = myRecorder.sample(batch_size, trace_length)
                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainN.predict, feed_dict={
                            mainN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
                            mainN.trainLength: trace_length, mainN.state_in: state_train, mainN.batch_size: batch_size})
                        Q2 = sess.run(targetN.Qout, feed_dict={
                            targetN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
                            targetN.trainLength: trace_length, targetN.state_in: state_train, targetN.batch_size: batch_size})
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size*trace_length), Q1]
                        targetQ = trainBatch[:, 2] + (y*doubleQ * end_multiplier)
                        # Update the network with our target values.
                        sess.run(mainN.updateModel,
                            feed_dict={mainN.scalarInput: np.vstack(trainBatch[:, 0]/255.0), mainN.targetQ: targetQ,
                            mainN.actions: trainBatch[:, 1], mainN.trainLength: trace_length,
                            mainN.state_in: state_train, mainN.batch_size: batch_size})
                rAll += r
                s = s1
                sP = s1P
                state = state1
                if d is True:
                    break

            # Add the episode to the episode recorder
            bufferArray = np.array(episodeBuffer)
            episodeBuffer = zip(bufferArray)
            myRecorder.add(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            print('Completed episode {0} with total score of {1}'.format(i, str(rAll)))

            # Periodically save the model.
            if i % 1000 == 0 and i != 0:
                saver.save(sess, path+'/model-'+str(i)+'.cptk')
                print("Saved Model")

        saver.save(sess, path+'/model-'+str(i)+'.cptk')


def main():
    parser = argparse.ArgumentParser(description="Train or run leaf classifier")
    parser.add_argument("-m", "--mode", help="Train / Run", required=True)
    args = vars(parser.parse_args())
    if args['mode'] == 'Train':
        train()
    else:
        print(':p Invalid Mode.')


if __name__ == "__main__":
    main()
