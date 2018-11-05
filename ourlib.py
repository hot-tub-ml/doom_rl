import numpy as np
from vizdoom import *
import time
import random
import skimage

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
    frame2 = skimage.transform.resize(frame, [84, 84])
#     plt.imshow(frame2);
    return frame2


def play_episode():
    """
        Play an episode by randomly choosing actions.
        Return pre-processed frames and action performed
        in each frame.
    """
    global game
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
#     game.close()
    return frames, labels
