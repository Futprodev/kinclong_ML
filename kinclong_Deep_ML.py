import cv2
import numpy as np
import os
import random
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
pgm_path = os.path.join(script_dir, "maps", "room1.pgm")

img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

_, binary_map = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)

binary_map = cv2.bitwise_not(binary_map)

contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

debug_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_img, contours, -1, (255, 255, 255), 2)

def points_in_contours(x, y, contours):
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            return True
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

for i in range(rows):
    for j in range(cols):
        x = j * grid_size + grid_size // 2
        y = i * grid_size + grid_size // 2
        if points_in_contours(x, y, contours):
            gridworld[i, j] = 1
        else:
            gridworld[i, j] = 0

gridworld = add_random_obstacle(gridworld, 5)

vis_grid = debug_img.copy()

for i in range(rows):
    for j in range(cols):
        x = j * grid_size
        y = i * grid_size
        color = (0, 0, 255) if gridworld[i, j] == 1 else (255, 0, 0)
        cv2.rectangle(vis_grid, (x, y), (x + grid_size, y + grid_size), color, 1)

resized_img = cv2.resize(vis_grid, (vis_grid.shape[1] * 2, vis_grid.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
resized_img_debug = cv2.resize(debug_img, (debug_img.shape[1] * 2, debug_img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)


cv2.imshow("Gridworld", resized_img_debug)
cv2.waitKey(0)
cv2.destroyAllWindows()