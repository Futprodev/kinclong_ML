import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import os
import random
import hashlib

style.use("ggplot")

HM_EPISODES = 16000

MOVE_PENALTY = 0.1
CLEAN_PENALTY = 3
CLEAN_REWARD = 30

epsilon = 1.0
EPSILON_RESET_INTERVAL = 4000
epsilon_reset_trigger = EPSILON_RESET_INTERVAL
MAX_EPSILON = 1.0
MIN_EPSILON = 0.2
STEP_SCALE = 1.2
EPS_DECAY = 0.9999
SHOW_EVERY = 2000

start_q_table = None
LEARNING_RATE = 0.05
DISCOUNT = 0.99

episode_rewards = []

VACCUUM_N = 1
DIRY_N = 2
CLEAN_N = 3

# pgm map conversion to use in opencv
script_dir = os.path.dirname(os.path.abspath(__file__))
pgm_path = os.path.join(script_dir, "maps", "room2.pgm")

img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

invert = cv2.bitwise_not(img)

_, binary_map = cv2.threshold(invert, 250, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

debug_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_img, contours, -1, (255, 255, 255), 2)

# Hash function for Q-table seeing the grid
def hash_state(posx, posy, tile_state):
    state_str = str((posx, posy, tile_state))
    return hashlib.md5(state_str.encode()).hexdigest()

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

def reward_for_lines(tile_grid, clean_value=3, base_reward=30):
    bonus = 0
    rows, cols = tile_grid.shape
    longest_row = 0
    longest_col = 0
    full_rows = []
    full_cols = []

    for i in range(rows):
        if all(tile_grid[i, j] == clean_value for j in range(cols)):
            bonus += base_reward * cols
            full_rows.append(i)
        else:
            streak = sum(1 for j in range(cols) if tile_grid[i, j] == clean_value)
            longest_row = max(longest_row, streak)

    for j in range(cols):
        if all(tile_grid[i, j] == clean_value for i in range(rows)):
            bonus += base_reward * rows
            full_cols.append(j)
        else:
            streak = sum(1 for i in range(rows) if tile_grid[i, j] == clean_value)
            longest_col = max(longest_col, streak)

    return bonus, longest_row, longest_col, full_rows, full_cols

def reward_for_quadrants(tile_grid, clean_value=3, base_reward=30):
    rows, cols = tile_grid.shape
    total_clean = (tile_grid == clean_value).sum()
    quarter_tiles = (rows * cols) // 4
    completed_quarters = total_clean // quarter_tiles
    bonus = completed_quarters * base_reward * 5
    return bonus, completed_quarters


# Color mapping: vacuum - orange, clean tile - green, dirty tile - red
d = {
    VACCUUM_N: (255, 175, 0),
    CLEAN_N: (0, 255, 0),
    DIRY_N: (0, 0, 255)
}

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

# Tiles class: initialize grid with dirty tiles
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
    
    def surroundings(self, bot):
        surrounding_tiles = []
        for dx in [-1 , 0 , 1]:
            for dy in [-1, 0, 1]:
                nx, ny = bot.x + dx, bot.y + dy
                if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                    surrounding_tiles.append(self.grid[ny][nx])
                else:
                    surrounding_tiles.append(-1) #invalid space (out of bounds)
        return tuple(surrounding_tiles)


if start_q_table is None:
    print("[INFO] Initializing new Q-table...")
    q_table = {}
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            q_table[(i, j)] = [np.random.uniform(-5, 0) for _ in range(4)]
else:
    print(f"[INFO] Loading Q-table from '{start_q_table}'...")
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)
    print(f"[INFO] Loaded Q-table with {len(q_table)} entries.")

for episode in range(HM_EPISODES):
    vac_bot = Bot()
    tile = Tiles()
    episode_reward = 0
    
    # Only compute the moving average if we have enough episodes.
    if episode % SHOW_EVERY == 0:
        if len(episode_rewards) >= SHOW_EVERY:
            avg_reward = np.mean(episode_rewards[-SHOW_EVERY:])
        else:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        print(f"on # {episode}, epsilon: {epsilon:.4f}")
        print(f"Last {SHOW_EVERY} episodes mean reward: {avg_reward}")
        
        show = True    
    else:
        show = False
    
    if episode != 0 and episode % epsilon_reset_trigger == 0:
        epsilon = min(epsilon + 0.3, MAX_EPSILON)
        steps = int(steps * STEP_SCALE)
        print(f"[SCHEDULER] Epsilon reset to {epsilon:.2f}, Steps increased to {steps}")
    
    # Make an observation on the bot and the surrounding tiles
    for step in range(steps):
        # this hash will be filled once the bot actually starts
        obs = hash_state(vac_bot.x, vac_bot.y, tile.surroundings(vac_bot))
        
        # we need to create a initial value for the start
        if obs not in q_table:
            q_table[obs] = [np.random.uniform(-5, 0) for _ in range(4)]

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        vac_bot.action(action)
        reward = tile.state(vac_bot)
        new_obs = hash_state(vac_bot.x, vac_bot.y, tile.surroundings(vac_bot))

        if new_obs not in q_table:
            q_table[new_obs] = [np.random.uniform(-5, 0) for _ in range(4)]

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q
        episode_reward += reward

        if show:
            # Display the current state of the grid.
            env = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)
            for row in range(SIZE_Y):
                for col in range(SIZE_X):
                    val = tile.grid[row][col]
                    if gridworld[row][col] == 0:
                        env[row][col] = (100, 100, 100)
                    elif tile.visit_count[row][col] > 3:
                        env[row][col] = (30, 30, 30)
                    else:
                        env[row][col] = d.get(val, (0, 0, 0))
                    
            env[vac_bot.y][vac_bot.x] = d[VACCUUM_N]
            tile_size = 20  # pixels per tile (adjust for your preference)

            # Resize entire map image before showing it
            upscaled_env = cv2.resize(env, (SIZE_X * tile_size, SIZE_Y * tile_size), interpolation=cv2.INTER_NEAREST)

            # create an outline of what the robot sees
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = vac_bot.x + dx, vac_bot.y +dy
                    if 0 <= nx < SIZE_X and 0 <= ny < SIZE_Y:
                        cv2.rectangle(upscaled_env, 
                                      (nx * tile_size, ny * tile_size),
                                      ((nx + 1) * tile_size, (ny + 1) * tile_size),
                                      (255, 255, 255), 2)

            cv2.imshow("Environment", upscaled_env)

            if cv2.waitKey(20) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                exit()
        

        if np.all(tile.grid == CLEAN_N):
            episode_reward += 0.5 * total_free_tiles * CLEAN_REWARD
            break

    lines_bonus, longest_row, longest_col, full_rows, full_cols = reward_for_lines(tile.grid)
    quad_bonus, completed_quarters = reward_for_quadrants(tile.grid)

    episode_reward += lines_bonus
    episode_reward += quad_bonus

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    if epsilon < 0.2:
        epsilon = 1.0

# Plot the moving average of rewards
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot(list(range(len(moving_avg))), moving_avg)
plt.ylabel(f"Reward ({SHOW_EVERY} episodes avg)")
plt.xlabel("Episode #")
plt.show()

print("\n[FINAL REPORT]")
print(f"[INFO] Longest cleaned row streak: {longest_row}")
print(f"[INFO] Longest cleaned column streak: {longest_col}")
print(f"[INFO] Fully cleaned row indices: {full_rows}")
print(f"[INFO] Fully cleaned column indices: {full_cols}")
print(f"[INFO] Completed {completed_quarters} quarter(s) of the map")
print(f"[INFO] Total line bonus: {lines_bonus}")
print(f"[INFO] Total quadrant bonus: {quad_bonus}")

# Save the Q-table
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
