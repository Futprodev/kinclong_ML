import numpy as np
# import keras.backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

# DQN PARAMETERS
LOAD_MODEL = None

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "kinclong_DQN"
MIN_BATCH_SIZE = 64
DISCOUNT = 0.99
UPDTAE_TARGET_EVERY = 5

MIN_REWARD = -200
MEMORY_FRACTION = 0.20

EPISODES = 1_000
epsilon = 1.0
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.2

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False

# Entity Selection
VACCUUM_N = 1
DIRTY_N = 2
CLEAN_N = 3

# PGM Map Params
GRID_SIZE = 10
PGM_PATH = None

# pgm map conversion
script_dir = os.path.dirname(os.path.abspath(__file__))
pgm_path = os.path.join(script_dir, "maps", "room2.pgm")

img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

invert = cv2.bitwise_not(img)

_, binary_map = cv2.threshold(invert, 250, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

debug_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_img, contours, -1, (255, 255, 255), 2)

def points_in_contours(x, y, contours, hierarchy):
    idx = 0
    while idx >= 0:
        if cv2.pointPolygonTest(contours[idx], (x, y), False) >= 0:
            if hierarchy[0][idx][3] != -1:
                return False
            else:
                return True
        idx = hierarchy[0][idx][0]
    return False

def add_random_obstacle(gridworld, num_obstacles):
    rows, cols = gridworld.shape
    free_cells = [(i, j) for i in range(rows) for j in range(cols) if gridworld[i, j] == 1]

    selected_cells = random.sample(free_cells, num_obstacles)
    for i, j in selected_cells:
        gridworld[i, j] = 0  # Mark as obstacle
    return gridworld

grid_size = 10
rows = img.shape[0] // grid_size
cols = img.shape[1] // grid_size

gridworld = np.zeros((rows, cols), dtype=np.uint8)
SIZE_Y, SIZE_X = gridworld.shape

for i in range(rows):
    for j in range(cols):
        x = j * grid_size + grid_size // 2
        y = i * grid_size + grid_size // 2
        if points_in_contours(x, y, contours, hierarchy):
            gridworld[i, j] = 1
        else:
            gridworld[i, j] = 0

gridworld = add_random_obstacle(gridworld, 0)

vis_grid = debug_img.copy()

for i in range(rows):
    for j in range(cols):
        x = j * grid_size
        y = i * grid_size
        color = (0, 255, 0) if gridworld[i, j] == 1 else (0, 0, 255)
        cv2.rectangle(vis_grid, (x, y), (x + grid_size, y + grid_size), color, 1)

class PGMroom:
    def __init__(self, pgm_path, gridsize):
        self.pgm_path = PGM_PATH
        self.gridsize = GRID_SIZE
        self.grid_world = None

    def generate_world(self):
        pgm_path = self.pgm_path

        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

        invert = cv2.bitwise_not(img)

        _, binary_map = cv2.threshold(invert, 250, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        debug_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, contours, -1, (255, 255, 255), 2)

        grid_size = self.gridsize
        rows = img.shape[0] // grid_size
        cols = img.shape[1] // grid_size

        gridworld = np.zeros((rows, cols), dtype=np.uint8)
        SIZE_Y, SIZE_X = gridworld.shape



# Skibidi the bot
class Bot:
    def __init__(self):
        while True:
            self.x = np.random.randint(0, SIZE_X)
            self.y = np.random.randitn(0, SIZE_Y)
            if gridworld[self.y][self.x]==1:
                break
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)


    def move(self, x=0, y=0):
        new_x = self.x + x
        new_y = self.y + y
        if 0 <= new_x < SIZE_Y and 0 <= new_y < SIZE_X:
            self.x = new_x
            self.y = new_y

# Creating the tiles
class Tiles:
    def __init___(self):
        self.grid = np.where(gridworld == 1, DIRTY_N, 0)

# Environment 
class Env:
    # think of a way to produce a image for the training of the gridworld and bot

    # Parameters
    MOVE_PENALTY = 0.1
    CLEAN_PENALTY = 3
    CLEAN_REWARD = 30

    RETURN_IMAGES = True
    OBSERVATION_SPACE_VALUES = (SIZE_X, SIZE_Y)
    ACTION_SPACE = 4

    d = {
        VACCUUM_N: (255, 175, 0),
        CLEAN_N: (0, 255, 0),
        DIRTY_N: (0, 0, 255)
    }



class ModifiedTensorBoard(tf.keras.callbaks.callback):
    def __init__(self, log_dir):
        super(ModifiedTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.write = tf.summary.create_file_writer(self.log_dir)
        self.step = 1
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            with self.writer.as_default():
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()
        self.step += 1

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()
    
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weight())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboar = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        
