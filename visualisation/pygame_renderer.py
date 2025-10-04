import numpy as np
import pygame
import os # For checking if image files exist

# Import config constants
from config import CELL_SIZE, FPS, COLOURS, GRID_MAPPING, ACTIONS

class PygameRenderer:
    def __init__(self, grid_height, grid_width, cell_size=CELL_SIZE):
        self.cell_size = cell_size
        self.window_width = grid_width * cell_size
        self.window_height = grid_height * cell_size
        self.screen = None

        self.images = {}
        self._load_assets() # Load all assets once

        self.curr_pacman_dir = 'right'
        self.last_player_pos = None

    def _load_assets(self):
        self.images[GRID_MAPPING['BLOCK']] = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
        self.images[GRID_MAPPING['BLOCK']].fill(COLOURS['BLACK'])

        # self.images[GRID_MAPPING['EMPTY']] = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
        # self.images[GRID_MAPPING['EMPTY']].fill(COLOURS['WHITE']) # Not strictly needed if cells are drawn white

        # Create a simple green square for the goal
        self.images[GRID_MAPPING['GOAL']] = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
        self.images[GRID_MAPPING['GOAL']].fill(COLOURS['GREEN'])


        # Load and scale actual image assets for ghost and pacman
        asset_path = os.path.join(os.path.dirname(__file__), '..', 'assets')

        ghost_raw = pygame.image.load(os.path.join(asset_path, 'ghost.png'))
        self.images[GRID_MAPPING['GHOST']] = pygame.transform.scale(ghost_raw, (self.cell_size - 8, self.cell_size - 8))

        pacman_raw = pygame.image.load(os.path.join(asset_path, 'pacman.png'))
        scaled_pacman = pygame.transform.scale(pacman_raw, (self.cell_size - 8, self.cell_size - 8))
        self.images['pacman'] = {
            'right': scaled_pacman,
            'up': pygame.transform.rotate(scaled_pacman, 90),
            'left': pygame.transform.rotate(scaled_pacman, 180),
            'down': pygame.transform.rotate(scaled_pacman, 270)
        }


    def init_display(self, caption="RL vs Pacman"):
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()

    def render(self, grid, player_pos=None):
        self.screen.fill(COLOURS['GRAY']) # Clear screen

        height, width = grid.shape

        # Determine Pac-Man's direction if player_pos is provided
        if player_pos and self.last_player_pos:
            dr, dc = player_pos[0] - self.last_player_pos[0], player_pos[1] - self.last_player_pos[1]
            if dr == -1: self.curr_pacman_dir = 'up'  # noqa: E701
            elif dr == 1: self.curr_pacman_dir = 'down'  # noqa: E701
            elif dc == -1: self.curr_pacman_dir = 'left'  # noqa: E701
            elif dc == 1: self.curr_pacman_dir = 'right'  # noqa: E701
        self.last_player_pos = player_pos


        for i in range(height):
            for j in range(width):
                # Draw cell background
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, COLOURS['WHITE'], rect)
                pygame.draw.rect(self.screen, COLOURS['BLACK'], rect, 1)  # Grid lines

                # Draw content based on cell value
                cell_value = grid[i, j]
                blit_pos = (j * self.cell_size + 2, i * self.cell_size + 2) # Default offset

                if cell_value == GRID_MAPPING['PLAYER']:
                    self.screen.blit(self.images['pacman'][self.curr_pacman_dir], (j * self.cell_size + 4, i * self.cell_size + 4))
                elif cell_value in self.images:
                    self.screen.blit(self.images[cell_value], blit_pos)

        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True # Signal to quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None, True # Signal to quit
                elif event.key == pygame.K_UP:
                    return ACTIONS['UP'], False
                elif event.key == pygame.K_DOWN:
                    return ACTIONS['DOWN'], False
                elif event.key == pygame.K_LEFT:
                    return ACTIONS['LEFT'], False
                elif event.key == pygame.K_RIGHT:
                    return ACTIONS['RIGHT'], False
        return None, False

    def close_display(self):
        """Quits pygame."""
        pygame.quit()