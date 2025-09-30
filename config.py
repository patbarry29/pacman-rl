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
    'GHOST': -10,
    'GOAL': 10,
    'STEP': -1
}

RANDOM_MOVE_PROB = 0

AGENT_PARAMS = {
    'alpha': 0.2,         # Learning rate
    'epsilon': 1.0,       # Initial exploration rate for training
    'min_epsilon': 0.005,  # Minimum exploration rate after decay
    'gamma': 0.9,         # Discount factor
    'decay': 0.005,       # Epsilon decay rate per episode
    'num_episodes': 300, # Total episodes for full training
    'max_steps_per_episode': 80, # Max steps an agent can take in one episode

    'num_episodes_for_viz_play': 200, # Fewer episodes for quick visualization during interactive play
    'num_episodes_for_quick_train': 500, # Episodes for generating a static final policy plot if no Q-table found
    'q_table_save_path': 'q_table.npy', # Default path for saving/loading Q-table (for single training run)
}

# Add BEST_Q_TABLE_PATH here. This is the path to your best trained Q-table
# You will update this manually after running run_experiments.py
BEST_Q_TABLE_PATH = 'experiment_results/q_tables/q_table_alpha_0p10_gamma_0p99_epsilon_start_1p00_decay_0p0010_avg30runs.npy'

# Environment parameters (already existing or added in previous step)
ENV_PARAMS = {
    'grid_filepath': "grid.npy",
}

# --- Parameters for Policy Evaluation --- # NEW SECTION
EVAL_PARAMS = {
    'num_eval_episodes': 1000, # Number of episodes to run for quantitative evaluation
    'max_steps_per_episode': 100, # Max steps per episode during evaluation (can be different from training)
    'render_evaluation': False, # Set to True to visualize evaluation episodes
}

# Expose common parameters for easier access in other modules
AGENT_RENDER_DELAY_MS = 200
# HUMAN_RENDER_DELAY_MS = PYGAME_SETTINGS['HUMAN_RENDER_DELAY_MS']

# EXPERIMENT_PARAMS is in config.py but not used in main.py anymore,
# it's used by run_experiments.py. Keep it in config.py for run_experiments.py
EXPERIMENT_PARAMS = {
    'alphas_to_test': [0.01, 0.1, 0.2],
    'gammas_to_test': [0.9, 0.99],
    'decays_to_test': [0.0001, 0.001, 0.005],
    'num_runs_per_param': 30, # How many times to repeat training for each param combination
}