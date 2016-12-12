Leaf Classification:
####
An application of deep reinforcement learning
---------

This model trains on grayscale images of 99 different species of leaves. Approximately 1580+ images in all and 16 images per species. For full description of the dataset see [kaggle](https://www.kaggle.com/c/leaf-classification/data).

Requirements:
----------------
- python 3.5
- [tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
- [keras](https://keras.io/#installation)
- [theano](http://deeplearning.net/software/theano/install.html)
- install remaining package dependencies via `pip install -r requirements.txt`

How to Train:
-----------------
- Clone this repo
- Download the [dataset](https://www.kaggle.com/c/leaf-classification/data)
- Navigate to repo directory and run `python preprocess.py -d 'your_path_to_data'`
	- This takes all leaf images stored in `your_path_to_data` and processes them to be 32x32 grayscale images.
	- Processed image files now located in the repo directory under `processed`
- Next navigate to whichever model you wish to train and run `python learn.py -m Train`

Model Descriptions:
---------------------
- Deep Recurrent Reinforcement Network
	- Located in `/reinforcement`
	- The model simulates a game in which the play has 99 possible moves/actions. 
	- Given an image of a leaf, the player must make one move. If the move matches with the leaf's species ID, then a positive reward is given. If not, a negative reward is given. This emulates a sort of "flash card" study game in which the learner looks at the image, makes a decision, and during training immediately discovers if the decision is accurate or not.
	- Model includes Long Short Term Memory (LSTM) components.
	- During training a target network and training network are used as a form of competitive learning.
- Deep Convolutional Neural Network (with Images)
	- Located in `/cnn`
	- Inputs: leaf image
	- Processes through two convolutional layers followed by two connected layers
	- Incorporates batch normalization, dropout regularization, and SGD
- Deep Convolutional Neural Network (with Feature Vectors)
	- Located in `/1d-nn`
	- Inputs: leaf features (margins, shapes, textures) formatted as 1-dimensional vector
	- Processes through two convolutional layers followed by two connected layers
	- Incorporates batch normalization, dropout regularization, and SGD
- Deep Highway Network
	- Located in `/highway-net`
	- Inputs: leaf image
	- More information soon
