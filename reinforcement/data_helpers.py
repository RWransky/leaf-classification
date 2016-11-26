import numpy as np
import random


def convert_species_to_labels(data):
    # create empty array to store new labels in
    labels = np.zeros((data.shape[0],))
    # find all unique species
    unique_species = np.unique(data)
    # label counter for assigning labels
    label_counter = 0
    # assign numberical labels for species
    for species in unique_species:
        ind = np.where(data == species)
        labels[ind] = label_counter
        label_counter += 1
    return labels


def convert_labels_to_species(data):
    # find all unique species
    unique_species = np.unique(data)
    # create empty list to store species
    species_list = []
    # assign species for numberical labels
    for species in unique_species:
        species_list.append(str(species))
    return species_list


def shuffle_data(data):
    data = data.reshape((data.shape[0],))
    num_samples = data.shape[0]
    # create empty array sized by 10 * num_samples
    shuffles = np.zeros((num_samples*100,))
    for i in range(100):
        shuffles[i*num_samples:(i+1)*num_samples] = random.sample(list(data), data.shape[0])
    return shuffles


def shuffle_test_data(data):
    data = data.reshape((data.shape[0],))
    num_samples = data.shape[0]
    # create empty array sized by 10 * num_samples
    shuffles = np.zeros((num_samples*1,))
    for i in range(1):
        shuffles[i*num_samples:(i+1)*num_samples] = random.sample(list(data), data.shape[0])
    return shuffles
