import os
import numpy as np
import random
import pandas as pd
from skimage import io

base_path = os.path.dirname(os.getcwd())


def load_data():
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
    images = convert_ids_to_images(image_id)
    return images, image_id, convert_labels_to_species()


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


def convert_ids_to_images(ids):
    num_ids = ids.shape[0]
    images = np.zeros((num_ids, 80, 80,), dtype=np.uint32)
    for i in range(num_ids):
        images[i] = load_image(ids[i])
    return images


def load_image(image_id):
    return io.imread('{0}/processed/{1}.jpg'.format(base_path, str(int(image_id))))
