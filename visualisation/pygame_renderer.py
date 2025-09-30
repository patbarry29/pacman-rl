import numpy as np
import pygame
import os # For checking if image files exist

# Import config constants
from config import CELL_SIZE, FPS, COLOURS, GRID_MAPPING, ACTIONS


# Initialize pygame
# pygame.init()

# # Grid visualization with Pygame
# def visualize_grid_pygame(grid, cell_size=None):
#     if cell_size is None:
#         cell_size = config.CELL_SIZE

#     colours = config.COLOURS

#     # Calculate window size
#     height, width = grid.shape
#     window_width = width * cell_size
#     window_height = height * cell_size

#     # Create window
#     screen = pygame.display.set_mode((window_width, window_height))
#     pygame.display.set_caption("RL vs Pacman")

#     # Load images (or create simple shapes)
#     ghost_img = pygame.image.load('ghost.png')
#     ghost_img = pygame.transform.scale(ghost_img, (cell_size-8, cell_size-8))
#     # ghost_img = pygame.Surface((cell_size-4, cell_size-4))
#     # ghost_img.fill(RED)
#     # pygame.draw.circle(ghost_img, (255, 255, 255), (cell_size//2, cell_size//2), cell_size//4)
#     pacman_img = pygame.image.load('pacman.png')

#     block_img = pygame.Surface((cell_size-4, cell_size-4))
#     block_img.fill(colours['BLACK'])
#     # pygame.draw.rect(block_img, BLACK, (5, 5, cell_size-14, cell_size-14))

#     empty_img = pygame.Surface((cell_size-4, cell_size-4))
#     empty_img.fill(colours['WHITE'])

#     # Current rotation angle (0 = right, 90 = up, 180 = left, 270 = down)
#     pacman_images = {
#         'right': pygame.transform.scale(pacman_img, (cell_size-8, cell_size-8)),
#         'up': pygame.transform.rotate(pygame.transform.scale(pacman_img, (cell_size-8, cell_size-8)), 90),
#         'left': pygame.transform.rotate(pygame.transform.scale(pacman_img, (cell_size-8, cell_size-8)), 180),
#         'down': pygame.transform.rotate(pygame.transform.scale(pacman_img, (cell_size-8, cell_size-8)), 270)
#     }
#     current_direction = 'right'

#     # Main game loop
#     running = True
#     clock = pygame.time.Clock()

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 pacman_pos = np.where(grid == 4)
#                 row = pacman_pos[0][0]
#                 col = pacman_pos[1][0]
#                 if event.key == pygame.K_ESCAPE:
#                     running = False
#                 elif event.key == pygame.K_UP:
#                     if row != 0:
#                         new_cell = grid[row-1][col]
#                         if new_cell == 1 or new_cell == 3:
#                             running = False
#                         elif new_cell == 0:
#                             grid[row-1][col] = 4
#                             grid[row][col] = 0
#                             current_direction = 'up'

#                 elif event.key == pygame.K_DOWN:
#                     if row != height-1:
#                         new_cell = grid[row+1][col]
#                         if new_cell == 1 or new_cell == 3:
#                             running = False
#                         elif new_cell == 0:
#                             grid[row+1][col] = 4
#                             grid[row][col] = 0
#                             current_direction = 'down'

#                 elif event.key == pygame.K_LEFT:
#                     if col != 0:
#                         new_cell = grid[row][col-1]
#                         if new_cell == 1 or new_cell == 3:
#                             running = False

#                         elif new_cell == 0:
#                             grid[row][col-1] = 4
#                             grid[row][col] = 0
#                             current_direction = 'left'

#                 elif event.key == pygame.K_RIGHT:
#                     if col != width-1:
#                         new_cell = grid[row][col+1]
#                         if new_cell == 1 or new_cell == 3:
#                             running = False
#                         elif new_cell == 0:
#                             grid[row][col+1] = 4
#                             grid[row][col] = 0
#                             current_direction = 'right'

#         # Clear screen
#         screen.fill(colours['GRAY'])

#         # Draw grid lines and cells
#         for i in range(height):
#             for j in range(width):
#                 # Draw cell background
#                 rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
#                 pygame.draw.rect(screen, colours['WHITE'], rect)
#                 pygame.draw.rect(screen, colours['BLACK'], rect, 1)  # Grid lines

#                 # Draw content based on cell value
#                 if grid[i, j] == 1:  # Ghost
#                     screen.blit(ghost_img, (j * cell_size + 2, i * cell_size + 2))
#                 elif grid[i, j] == 2:  # Block
#                     screen.blit(block_img, (j * cell_size + 2, i * cell_size + 2))
#                 elif grid[i, j] == 3:  # Goal
#                     goal_img = pygame.Surface((cell_size-4, cell_size-4))
#                     goal_img.fill(colours['GREEN'])
#                     screen.blit(goal_img, (j * cell_size + 2, i * cell_size + 2))
#                 elif grid[i, j] == 4:  # Player (you can add this later)
#                     screen.blit(pacman_images[current_direction], (j * cell_size + 4, i * cell_size + 4))
#                     # pygame.draw.circle(screen, YELLOW,
#                     #                  (j * cell_size + cell_size//2, i * cell_size + cell_size//2),
#                     #                 cell_size//3)

#         # Update display
#         pygame.display.flip()
#         clock.tick(config.FPS)  # 60 FPS

#     pygame.quit()






    # visualization/pygame_renderer.py

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