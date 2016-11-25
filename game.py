import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage import io

import data_helpers


class GameState:
    def __init__(self):
        self.score = self.loopIter = 0
        self.frames, self.labels, self.ids = load_data()

    def reset(self):
        self.score = self.loopIter = 0
        self.frames, self.labels, self.ids = load_data()
        ind_label = np.where(self.ids == int(self.frames[self.loopIter]))
        return load_image(self.frames[self.loopIter]), self.labels[ind_label[0]]

    def frame_step(self, input_action):
        ind_label = np.where(self.ids == int(self.frames[self.loopIter]))
        if self.labels[ind_label[0]] == input_action:
            self.score += 1
            reward = 1
        else:
            self.score -= 1
            reward = -1

        self.loopIter += 1
        image_data = load_image(self.frames[self.loopIter])
        ind_label = np.where(self.ids == int(self.frames[self.loopIter]))
        return image_data, reward, False, self.labels[ind_label[0]]


class TestState:
    def __init__(self):
        self.loopIter = 0
        self.frames = load_test_data()

    def reset(self):
        self.loopIter = 0
        self.frames = load_test_data()
        image_id = int(self.frames[self.loopIter])
        return load_image(self.frames[self.loopIter]), self.frames.shape[0], image_id, load_species_list()

    def frame_step(self, input_action):
        self.loopIter += 1
        image_data = load_image(self.frames[self.loopIter])
        image_id = int(self.frames[self.loopIter])
        return image_data, image_id


def load_test_data():
    df = pd.read_csv('test.csv')
    image_id = df[['id']].values
    shuffle = data_helpers.shuffle_test_data(image_id)
    return shuffle


def load_species_list():
    df = pd.read_csv('train.csv')
    return data_helpers.convert_labels_to_species(df.species)


def load_data():
    df = pd.read_csv('train.csv')
    image_id = df[['id']].values
    species = df.species
    labels = data_helpers.convert_species_to_labels(species)
    shuffle = data_helpers.shuffle_data(image_id)
    return shuffle, labels, image_id


def load_image(image_id):
    return io.imread('processed/{}.jpg'.format(str(int(image_id))))
