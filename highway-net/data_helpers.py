import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage import io

base_path = os.path.dirname(os.getcwd())


def load_data():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    image_id = df[['id']].values
    species = df.species
    labels = convert_species_to_labels(species)
    images = convert_ids_to_images(image_id)
    return images, labels


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


def convert_ids_to_images(ids):
    num_ids = ids.shape[0]
    images = np.zeros((num_ids, 80, 80,), dtype=np.uint32)
    for i in range(num_ids):
        images[i] = load_image(ids[i])
    return images


def load_image(image_id):
    return io.imread('{0}/processed/{1}.jpg'.format(base_path, str(int(image_id))))
