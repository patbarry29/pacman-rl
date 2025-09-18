import numpy as np
import config

def load_grid(filepath):
    grid = np.load(filepath)

    grid_mapping = config.GRID_MAPPING

    start_pos = np.where(grid==grid_mapping['PLAYER'])
    start_pos = (start_pos[0][0], start_pos[1][0])

    goal_pos = np.where(grid==grid_mapping['GOAL'])
    goal_pos = (goal_pos[0][0], goal_pos[1][0])

    ghost_positions = np.where(grid==config.GRID_MAPPING['GHOST'])
    ghost_positions = [(int(r), int(c)) for r, c in zip(ghost_positions[0], ghost_positions[1])]

    # print(grid)

    return grid, start_pos, goal_pos, ghost_positions