import numpy as np
from vizdoom import *
import time
import random
import skimage
from tensorflow.keras import layers
from tensorflow.keras import models

game = DoomGame()  # must be global

# Possible actions
left = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]
possible_actions = [left, right, shoot]
# `repr` because lists cannot be keys in a dict
action_dict = {repr(left):'L', repr(right):'R', repr(shoot):'S'} 

def init_basic_env():
    """
        Load basic.cfg and basic.wad
    """
    global game
    game.load_config("./basic.cfg")
    game.set_doom_scenario_path("./basic.wad")
    game.init()

def preprocess_frame(img):
    """
        Apply pre-processing to `img` (screen buffer).
        Convert from RGB to greyscale, crop ceiling and bar,
        resize it to 84x84, standardize pxs.
    """
    img = np.transpose(img, (1, 2, 0))
    img_grey = skimage.color.rgb2grey(img)
    frame = img_grey
#     plt.imshow(img_grey);
    frame = frame[50:200, :]
    frame /= 255
#     plt.imshow(frame);
    frame2 = skimage.transform.resize(frame, [48, 48])
#     plt.imshow(frame2);
    return frame2


def play_episode():
    """
        Play an episode by randomly choosing actions.
        Return pre-processed frames and action performed
        in each frame.
    """
    global game
    frames = []
    while len(frames) < 24:  # repeat if episode is too short
        game.new_episode()
        frames = []
        labels = []
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            frame = preprocess_frame(img)
            action = random.choice(possible_actions)
            labels.append(action_dict[repr(action)])
            reward = game.make_action(action)
            time.sleep(10 ** -5)
            frames.append(frame)
    return frames, labels

def stack_frames(frames, history_length=6):
    stacks = [np.hstack(frames[ix:ix+history_length]) \
              for ix in range(len(frames)- history_length)]
    stacks = np.asarray(stacks)
    # new_shape = stacks.shape + (1,) # add fourth dummy dim
    # stacks = np.reshape(stacks, new_shape)
    return stacks


def f2a_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 288, 1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def get_images_labels(number, history_length=6):
    """
        Get at least `number` of images and labels 
        by playing as many episodes as needed.
    """
    trn_images = None
    trn_labels = None
    while True:
        frames, labels = play_episode()
        stacks = stack_frames(frames, history_length)
        labels = labels[:-history_length]
        if trn_images is None or trn_labels is None:
            trn_images = stacks
            trn_labels = labels
        else:
            trn_images = np.append(trn_images, stacks, axis=0)
            trn_labels = np.append(trn_labels, labels)
        if len(trn_labels) > number:
            break
    # add fourth dummy dim to images
    new_shape = trn_images.shape + (1,) # add fourth dummy dim
    trn_images = np.reshape(trn_images, new_shape)
    return trn_images, trn_labels

def oh_labels(labels):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    le = LabelEncoder()
    oh = OneHotEncoder()
    labels_le = le.fit_transform(labels)
    labels_le = np.reshape(labels_le, (-1, 1))
    labels_le.shape
    labels_oh = oh.fit_transform(labels_le)
    return labels_oh