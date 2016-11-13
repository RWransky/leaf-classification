import os
import numpy as np
import random
import pandas as pd
from skimage import io

from data_helpers import *


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.frames, self.labels = self.load_data()

    def frame_step(self, input_actions, do_nothing=False):
        if !do_nothing:
            reward = 0.1
            terminal = False

            if sum(input_actions) != 1:
                raise ValueError('Multiple input actions!')

            # input_actions[x] == 1: id as species x
            ind = np.where(input_actions == 1)
            if self.labels[self.frames[self.loopIter]] == ind:
                self.score += 1
                reward = 1
                self.loopIter += 1
            else:
                terminal = True
                self.__init__()
                reward = -1
        else:
            reward = 0.1
            terminal = False

        image_data = load_image(self.frames[self.loopIter])

        return image_data, reward, terminal

    def load_data():
        image_id = df[['id']].values
        species = df.species
        labels = data_helpers.convert_species_to_labels(species)
        shuffle = data_helpers.shuffle_data(image_id)
        return shuffle, labels

    def load_image(image_id):
        return io.imread('processed/{}.jpg'.format(str(image_id)))
