import numpy as np

def add_elements(grid, num_elements, value):
    elements = set()
    while len(elements) < num_elements:
        row = np.random.randint(0, grid.shape[0])
        col = np.random.randint(0, grid.shape[1])
        # Only add if the cell is empty (0)
        if grid[row, col] == 0:
            elements.add((row, col))

    elements = np.array(list(elements))
    grid[elements[:, 0], elements[:, 1]] = value
    return grid

def make_grid(N, M, num_ghosts, block_ratio):
    grid = np.zeros((N, M))

    # add ghosts
    grid = add_elements(grid, num_ghosts, 1)

    # add blocks
    num_blocks = int(round(block_ratio * (N * M)))
    grid = add_elements(grid, num_blocks, 2)

    # add target
    grid = add_elements(grid, 1, 3)

    # pacman
    grid = add_elements(grid, 1, 4)

    return grid