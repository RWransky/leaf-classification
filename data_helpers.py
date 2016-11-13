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


def shuffle_data(data):
    data = data.reshape((data.shape[0],))
    num_samples = data.shape[0]
    # create empty array sized by 10 * num_samples
    shuffles = np.zeros((num_samples*10,))
    for i in range(10):
        shuffles[i*num_samples:(i+1)*num_samples] = random.sample(list(data), data.shape[0])
    return shuffles
