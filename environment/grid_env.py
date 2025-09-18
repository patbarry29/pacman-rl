# environment/grid_env.py
import numpy as np
import random
from config import GRID_MAPPING, ACTIONS, REWARDS, RANDOM_MOVE_PROB # Assume REWARDS and RANDOM_MOVE_PROB are in config

class GridEnv:
    def __init__(self, base_grid_array, start_pos, goal_pos, ghost_positions):
        self.base_grid = np.copy(base_grid_array) # Store original immutable grid definition
        self.height, self.width = self.base_grid.shape
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.ghost_positions = ghost_positions

        self.curr_grid = None
        self.player_pos = None

        self.action_map = {
            ACTIONS['UP']: (-1, 0),
            ACTIONS['DOWN']: (1, 0),
            ACTIONS['LEFT']: (0, -1),
            ACTIONS['RIGHT']: (0, 1),
        }

        self.reset()


    def reset(self):
        self.curr_grid = np.copy(self.base_grid)
        self.player_pos = self.start_pos
        # The state for Q-learning is just the player's position
        return self.player_pos

    def get_valid_actions(self, state):
        r, c = state
        valid_actions = []
        for action_name, action_int in ACTIONS.items():
            dr, dc = self.action_map[action_int]
            new_r, new_c = r + dr, c + dc

            # Check bounds
            if 0 <= new_r < self.height and 0 <= new_c < self.width:
                # Check for walls (blocks)
                if self.curr_grid[new_r, new_c] != GRID_MAPPING['BLOCK']:
                    valid_actions.append(action_int)
        return valid_actions

    def step(self, action_input):
        current_r, current_c = self.player_pos

        # --- Apply Randomness ---
        # if random.random() < RANDOM_MOVE_PROB:
        #     valid_moves = self.get_valid_actions(self.player_pos)
        #     if valid_moves: # Ensure there are valid moves to pick from
        #         action = random.choice(valid_moves)
        #     else: # If no valid moves, just stick to original action or stay
        #         action = action_input
        # else:
        action = action_input
        # --- End Randomness ---

        dr, dc = self.action_map[action]
        new_r, new_c = current_r + dr, current_c + dc

        reward = REWARDS['STEP'] # Default step reward
        done = False

        # Check for boundaries and blocks
        if not (0 <= new_r < self.height and 0 <= new_c < self.width) or \
            self.curr_grid[new_r, new_c] == GRID_MAPPING['BLOCK']:
            # Invalid move, player stays in current position
            new_r, new_c = current_r, current_c
        else:
            new_player_pos = (new_r, new_c)

            self.curr_grid[current_r, current_c] = GRID_MAPPING['EMPTY'] # Clear old position
            self.curr_grid[new_player_pos] = GRID_MAPPING['PLAYER']     # Set new position

            self.player_pos = new_player_pos

            # Check for ghosts
            if self.player_pos in self.ghost_positions:
                reward = REWARDS['GHOST']
                done = True
            # Check for goal
            elif self.player_pos == self.goal_pos:
                reward = REWARDS['GOAL']
                done = True

        next_state = self.player_pos

        return next_state, reward, done

    def get_grid_snapshot(self):
        """Returns a numpy array representing the current visual state of the grid."""
        display_grid = np.copy(self.curr_grid)
        # Mark ghosts (they are static, but explicit is good)
        for r, c in self.ghost_positions:
            display_grid[r, c] = GRID_MAPPING['GHOST']
        # Mark the goal
        display_grid[self.goal_pos] = GRID_MAPPING['GOAL']
        # Mark the player's current position
        display_grid[self.player_pos] = GRID_MAPPING['PLAYER']
        return display_grid

    def get_player_direction(self, old_pos, new_pos):
        """Determines player direction for visualization based on movement."""
        if old_pos is None or old_pos == new_pos:
            return 'right' # Default if no movement or first frame

        dr, dc = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
        if dr == -1: return 'up'  # noqa: E701
        if dr == 1: return 'down'  # noqa: E701
        if dc == -1: return 'left'  # noqa: E701
        if dc == 1: return 'right'  # noqa: E701
        return 'right' # Fallback