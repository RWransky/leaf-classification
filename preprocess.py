import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
from os import listdir, makedirs
from os.path import isfile, join, exists


def get_images(file_dir):
    return [f for f in listdir(file_dir) if isfile(join(file_dir, f))]


def load_image(fname):
    return io.imread(fname)


def save_image(array, fname, directory='processed'):
    if not exists(directory):
        makedirs(directory)
    io.imsave('processed/{}'.format(fname), array)


def pad_image(array):
    w, h = array.shape
    # find max side length
    max_side = np.max((w, h))
    # create empty  square array
    square = np.zeros((max_side, max_side))
    # find indices to center array
    ind_w = int((max_side-w)/2)
    ind_h = int((max_side-h)/2)
    # place original array in center
    square[ind_w:ind_w+w, ind_h:ind_h+h] = array
    # return padded array
    return square


def scale_image(array):
    return cv2.resize(np.uint8(array), (80, 80))


def preprocess(args):
    file_dir = args['file_dir']

    image_files = get_images(file_dir)
    for file in image_files:
        array = load_image('{0}/{1}'.format(file_dir, file))
        padded = pad_image(array)
        scaled = scale_image(padded)
        save_image(scaled, file)


def main():
    parser = argparse.ArgumentParser(description='Script to preprocess leaf images')
    parser.add_argument('-d', '--file_dir', help='Directory where leaf images stored', required=True)
    args = vars(parser.parse_args())
    preprocess(args)

if __name__ == "__main__":
    main()
