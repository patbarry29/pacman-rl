CELL_SIZE = 50

FPS = 60

COLOURS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (200, 200, 200),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (210, 210, 0)
}

GRID_MAPPING = {
    'EMPTY': 0,
    'GHOST': 1,
    'BLOCK': 2,
    'GOAL': 3,
    'PLAYER': 4
}

ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3
}

REWARDS = {
    'GHOST': -5,
    'GOAL': 5,
    'STEP': -1
}

RANDOM_MOVE_PROB = 0

AGENT_PARAMS = {
    'alpha': 0.1,         # Learning rate
    'epsilon': 1.0,       # Initial exploration rate for training
    'min_epsilon': 0.01,  # Minimum exploration rate after decay
    'gamma': 0.9,         # Discount factor
    'decay': 0.005,       # Epsilon decay rate per episode
    'num_episodes': 1000, # Total episodes for full training
    'max_steps_per_episode': 200, # Max steps an agent can take in one episode

    'num_episodes_for_viz_play': 200, # Fewer episodes for quick visualization during interactive play
    'num_episodes_for_final_plot': 500, # Episodes for generating a static final policy plot
    'q_table_save_path': 'q_table.npy', # Default path for saving/loading Q-table
}


# Expose common parameters for easier access in other modules
AGENT_RENDER_DELAY_MS = 100
# HUMAN_RENDER_DELAY_MS = PYGAME_SETTINGS['HUMAN_RENDER_DELAY_MS']
