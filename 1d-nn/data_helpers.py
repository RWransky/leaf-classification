import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage import io
from sklearn.utils import shuffle
from scipy import ndimage

base_path = os.path.dirname(os.getcwd())


def load_data():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    image_id = df[['id']].values
    species = df.species
    species = species.values.reshape((species.shape[0], 1))
    stacked = np.concatenate((image_id, species), axis=1)
    image_id, labels = convert_species_to_labels(stacked)
    size = labels.shape[0]
    margins = pull_values(df, 'margin', size)
    shapes = pull_values(df, 'shape', size)
    textures = pull_values(df, 'texture', size)
    return margins, shapes, textures, labels


def load_full_data():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    image_id = df[['id']].values
    species = df.species
    species = species.values.reshape((species.shape[0], 1))
    stacked = np.concatenate((image_id, species), axis=1)
    image_id, labels = convert_species_to_labels(stacked)
    images = convert_ids_to_images(image_id)
    return images, labels


def load_test_data():
    df = pd.read_csv('{}/test.csv'.format(base_path))
    image_id = df[['id']].values
    size = image_id.shape[0]
    margins = pull_values(df, 'margin', size)
    shapes = pull_values(df, 'shape', size)
    textures = pull_values(df, 'texture', size)
    features = combine_features(margins, shapes, textures)
    return features, image_id, convert_labels_to_species()


def pull_values(df, column_type, size):
    data = np.zeros((size, 64))
    for i in range(64):
        data[:, i] = (df[['{0}{1}'.format(column_type, str(i+1))]].values).reshape((size,)).astype(np.float32)
    return data


def combine_features(m, s, t):
    intermediate = np.concatenate((m, s), axis=1)
    return np.concatenate((intermediate, t), axis=1)


def augment_data(images, labels):
    more_images = np.zeros((8*images.shape[0], images.shape[1], images.shape[2],))
    more_labels = np.zeros((8*labels.shape[0]))
    for i in range(labels.shape[0]):
        for j in range(8):
            rotation = j*90
            more_images[8*i+j] = ndimage.rotate(images[i], rotation)
            more_labels[8*i+j] = labels[i]
    return shuffle(more_images, more_labels)


def augment_test_data(images):
    more_images = np.zeros((4*images.shape[0], images.shape[1], images.shape[2],))
    for i in range(images.shape[0]):
        for j in range(4):
            rotation = j*90
            more_images[4*i+j] = ndimage.rotate(images[i], rotation)
    return more_images


def convert_species_to_labels(data):
    # create empty array to store new labels in
    labels = np.zeros((data.shape[0],))
    # create empty array to store image ID's in
    image_id = np.zeros((data.shape[0],))
    # find all unique species
    unique_species = np.unique(data[:, 1])
    # label counter for assigning labels
    label_counter = 0
    # assign numberical labels for species
    for species in unique_species:
        ind = np.where(data[:, 1] == species)
        labels[ind] = label_counter
        image_id[ind] = data[ind, 0]
        label_counter += 1
    return image_id, labels


def convert_labels_to_species():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    data = df.species
    # find all unique species
    unique_species = np.unique(data)
    # create empty list to store species
    species_list = []
    # assign species for numberical labels
    for species in unique_species:
        species_list.append(str(species))
    return species_list
