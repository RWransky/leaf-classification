import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage import io

import data_helpers


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.frames, self.labels, self.indices = load_data()

    def frame_step(self, input_actions, do_nothing=False):
        if not do_nothing:
            reward = 0.1
            terminal = False

            if sum(input_actions) != 1:
                raise ValueError('Multiple input actions!')

            # input_actions[x] == 1: id as species x
            ind_action = np.where(input_actions == 1)
            ind_label = np.where(self.indices == int(self.frames[self.loopIter]))
            if self.labels[ind_label[0]] == ind_action:
                self.score += 1
                reward = 1
                self.loopIter += 1
            else:
                self.score -= 1
                self.loopIter += 1
                if self.score < -5:
                    terminal = True
                    self.__init__()
                    reward = -1
                else:
                    terminal = False
                    reward = -1
        else:
            reward = 0.1
            terminal = False

        image_data = load_image(self.frames[self.loopIter])

        return image_data, reward, terminal


def load_data():
        df = pd.read_csv('train.csv')
        image_id = df[['id']].values
        species = df.species
        labels = data_helpers.convert_species_to_labels(species)
        shuffle = data_helpers.shuffle_data(image_id)
        return shuffle, labels, image_id


def load_image(image_id):
        return io.imread('processed/{}.jpg'.format(str(int(image_id))))
