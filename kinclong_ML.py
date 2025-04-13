import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 5
HM_EPISODES = 200

MOVE_PENALTY = 0.1
CLEAN_PENALTY = 2 
CLEAN_REWARD = 10

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 50

start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95

episode_rewards = []
steps = 200

VACCUUM_N = 1
DIRY_N = 2
CLEAN_N = 3

# Color mapping: vacuum - orange, clean tile - green, dirty tile - red
d = {
    VACCUUM_N: (255, 175, 0),
    CLEAN_N: (0, 255, 0),
    DIRY_N: (0, 0, 255)
}

# Bot class: randomized start position
class Bot:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
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
        if 0 <= self.x + x < SIZE:
            self.x += x
        if 0 <= self.y + y < SIZE:
            self.y += y

# Tiles class: initialize grid with dirty tiles
class Tiles:
    def __init__(self):
        # Initialize grid with DIRY_N (2) for dirty tiles
        self.grid = np.full((SIZE, SIZE), DIRY_N, dtype=int)

    def state(self, bot):
        # Use bot.y for row and bot.x for column
        current_value = self.grid[bot.y][bot.x]
        if current_value == DIRY_N:
            self.grid[bot.y][bot.x] = CLEAN_N
            return CLEAN_REWARD
        elif current_value == CLEAN_N:
            return -CLEAN_PENALTY
        else:
            return -MOVE_PENALTY

# Initialize Q-table
if start_q_table is None:
    q_table = {}
    for i in range(SIZE):
        for j in range(SIZE):
            q_table[(i, j)] = [np.random.uniform(-5, 0) for _ in range(4)]
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

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
    
    for step in range(steps):
        obs = (vac_bot.x, vac_bot.y)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        vac_bot.action(action)
        reward = tile.state(vac_bot)
        new_obs = (vac_bot.x, vac_bot.y)

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == CLEAN_REWARD:
            new_q = reward
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q
        episode_reward += reward

        if show:
            # Display the current state of the grid.
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            for row in range(SIZE):
                for col in range(SIZE):
                    env[row][col] = d[tile.grid[row][col]]
            # Place the bot's visual in its position. Note: using y,x order.
            env[vac_bot.y][vac_bot.x] = d[VACCUUM_N]
            img_disp = Image.fromarray(env, 'RGB')
            img_disp = img_disp.resize((300, 300), resample=Image.NEAREST)
            cv2.imshow("Environment", np.array(img_disp))
            # Optionally remove extra key checking if you don't want to interrupt

            if cv2.waitKey(125) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                exit()
        

        if np.all(tile.grid == CLEAN_N):
            episode_reward += 250
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    if epsilon < 0.1:
        epsilon = 0.1

# Plot the moving average of rewards
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot(list(range(len(moving_avg))), moving_avg)
plt.ylabel(f"Reward ({SHOW_EVERY} episodes avg)")
plt.xlabel("Episode #")
plt.show()

# Save the Q-table
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
