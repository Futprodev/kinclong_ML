import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 5
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 100
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95
episode_rewards = []
episode_reward = 0
steps = 200

# Entity Dictionary
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# Set entity colours BGR
d = {
    1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
}

# Create Blob class for handling the behaviour of entities
class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"
    
    # to remove another blob
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    # blob action, improve by making it able to move in x and y also
    def action(self, choice):
        # if choice == 0:
        #     self.move(x=1, y=1)       # →↓ (diagonal)
        # elif choice == 1:
        #     self.move(x=-1, y=-1)     # ←↑ (diagonal)
        # elif choice == 2:
        #     self.move(x=-1, y=1)      # ←↓ (diagonal)
        # elif choice == 3:
        #     self.move(x=1, y=-1)      # →↑ (diagonal)
        
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)

    def move(self, x=False, y=False):
        # if the blob hits the window border
        if 0 <= self.x + x < SIZE:
            self.x += x
        if 0 <= self.y + y < SIZE:
            self.y += y

if start_q_table is None:
    q_table = {}

    # (x1,y1) (x2,y2) delta of player to the enemy
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1,y1),(x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]

else:
    # load a pretrained model if there is one
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True    
    else:
        show = False

    # setting the action based on the food and enemy
    for i in range(steps):
        # the delta position
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        # optional
        # enemy.move()
        # food.move()

        # setting the reward
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
    
        # make a new observation after a move
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
        
        # make the image environment
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            # in an array y,x not x,y
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((450, 450), resample=Image.NEAREST)

            cv2.imshow("", np.array(img))

            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

    if epsilon < 0.1:
        epsilon = 0.1

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
