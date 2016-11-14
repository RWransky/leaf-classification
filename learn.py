'''
Leaf Classifier via Deep Reinforcement Learning

Initial code inspired by @yanpanlau

Extensive changes to game model and training methods

'''

from __future__ import print_function

import game

import argparse

import random
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
# the name of the game being played for log files
GAME = "leaf"
CONFIG = "nothreshold"
# number of valid actions
ACTIONS = 99
# decay rate of past observations
GAMMA = 0.99
# timesteps to observe before training
OBSERVATION = 100000.
# frames over which to anneal epsilon
EXPLORE = 98000.
# final value of epsilon
FINAL_EPSILON = 0.0001
# starting value of epsilon
INITIAL_EPSILON = 0.1
# number of previous transitions to remember
REPLAY_MEMORY = 50000
# size of minibatch
BATCH = 32
FRAME_PER_ACTION = 1

img_rows, img_cols = 80, 80
# image is binary
img_channels = 1


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape,
        name: normal(shape, scale=0.01, name=name), border_mode='same',
        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape,
        name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape,
        name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(model, args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and get the image
    a_t = np.zeros((ACTIONS,))
    s_t, r_0, terminal = game_state.frame_step(a_t, do_nothing=True)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, 1, s_t.shape[0], s_t.shape[1])

    if args['mode'] == 'Run':
        # We keep observe, never train
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        print ("Weights loaded successfully")
    else:
        # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    while (t < EXPLORE+OBSERVE-1):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                # input image, get the prediction
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # We reduce the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.frame_step(a_t)

        s_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            # 32, 1, 80, 80
            inputs = np.zeros((BATCH, 1, s_t.shape[2], s_t.shape[3]))
            print(inputs.shape)
            # 32, 99
            targets = np.zeros((inputs.shape[0], ACTIONS))

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                # This is action index
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i+1] = state_t
                # Choosing certain species probability
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t += 1

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # save progress every 100 iterations and if training
        if (t % 100 == 0) & (state == "train"):
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame(args):
    model = buildmodel()
    trainNetwork(model, args)


def main():
    parser = argparse.ArgumentParser(description="Train or run leaf classifier")
    parser.add_argument("-m", "--mode", help="Train / Run", required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
