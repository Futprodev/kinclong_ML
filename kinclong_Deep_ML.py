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

LOAD_MODEL = None

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "256x2"
MINI_BATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 1_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# pgm map conversion to use in opencv
script_dir = os.path.dirname(os.path.abspath(__file__))
pgm_path = os.path.join(script_dir, "maps", "room2.pgm")

# Map Setup
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

total_free_tiles = np.sum(gridworld == 1)
steps = int(4 * total_free_tiles)

# Bot class: randomized start position
class Bot:
    def __init__(self): 
        while True:
            self.x = np.random.randint(0, SIZE_X)
            self.y = np.random.randint(0, SIZE_Y)
            if gridworld[self.y][self.x] == 1:
                break
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=-1)
        elif choice == 3:
            self.move(x=0, y=1)

    def move(self, x=0, y=0):  
        new_x = self.x + x
        new_y = self.y + y
        if 0 <= new_x < SIZE_Y and 0 <= new_y < SIZE_X:
            if gridworld[new_y][new_x] == 1:
                self.x = new_x
                self.y = new_y

class Tiles:
    def __init__(self):
        # Initialize grid with DIRY_N (2) for dirty tiles
        self.grid = np.where(gridworld == 1, DIRY_N, 0)  # Set valid positions to DIRY_N
        self.visit_count = np.zeros_like(self.grid, dtype=int)  # Initialize visit count for each tile

    def state(self, bot):
        self.visit_count[bot.y][bot.x] += 1  # Increment visit count for the current tilr
        
        # Use bot.y for row and bot.x for column
        current_value = self.grid[bot.y][bot.x]
        if current_value == DIRY_N:
            self.grid[bot.y][bot.x] = CLEAN_N
            return CLEAN_REWARD
        elif current_value == CLEAN_N:
            revisit_penalty = CLEAN_PENALTY * min(self.visit_count[bot.y][bot.x], 3)  # Cap the penalty to avoid excessive negative rewards
            return -revisit_penalty
        else:
            return -MOVE_PENALTY
        
class RoomEnv:
    RETURN_IMAGES = True
    MOVE_PENALTY = 0.1
    CLEAN_PENALTY = 30
    CLEAN_REWARD = 10
    ACTION_SPACE_SIZE = 4

    VAC_BOT_N = 1
    CLEAN_N = 2
    DIRTY_n = 3

    d = {
        1: (255, 175, 0),
        2: (0, 255, 0),
        3: (0, 0, 255)
    }

    def reset(self):
        self.
        


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class ModifiedTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(ModifiedTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
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

        # main model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])


        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, 
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def create_model(self):
        
        if LOAD_MODEL is not None:
            print(f"Loading model {LOAD_MODEL}...")
            model = tf.keras.models.load_model(LOAD_MODEL)
            return model
        else:
            model = Sequential()
            model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3,3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

        return model

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    episode_reward = 0
    step = 1
    done = False
    current_state = env.reset()

    while not done:
        # Exploration vs Exploitation
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)
        episode_reward += reward

        # If not terminal state, add to replay memory and train main model
        if not done:
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(terminal_state=False, step=step)
        else:
            # If terminal state, train last time and set terminal state flag to True
            agent.train(terminal_state=True, step=step)

        current_state = new_state

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

    # Decay epsilon every episode
    if MIN_EPSILON < epsilon > 0.001:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    ep_rewards.append(episode_reward)

    # Stats for plotting
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        print(f"Episode: {episode}, "
              f"Average Reward: {average_reward}, "
              f"Min Reward: {min_reward}, "
              f"Max Reward: {max_reward}, "
              f"Epsilon: {epsilon}")

        # Save model every 10% of episodes
        if average_reward >= MIN_REWARD:
            agent.model.save(f"models/{MODEL_NAME}__{average_reward}__{int(time.time())}.model")