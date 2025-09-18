import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import config
from environment.grid_env import GridEnv

def plot_policy_frame(ax, values, actions, env, rewards, episode_num=None):
    """
    Plots a single frame of the Q-table policy and values as a heatmap with arrows.
    """
    vmin, vmax = min(rewards.values()), max(rewards.values())

    block_value = config.GRID_MAPPING['BLOCK']
    block_coords = np.where(env.base_grid == block_value)

    # Create a mask for wall cells
    mask = np.zeros_like(values, dtype=bool)
    for r, c in zip(block_coords[0], block_coords[1]):
        mask[r, c] = True
    vals_masked = np.ma.masked_array(values, mask)

    # Heatmap
    im = ax.imshow(vals_masked, cmap="coolwarm", origin="upper", vmin=vmin, vmax=vmax)

    # Arrows for best action
    for r in range(env.height):
        for c in range(env.width):
            if mask[r, c]:
                continue

            # Action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
            a = actions[r, c]
            if a == config.ACTIONS['UP']:
                ax.arrow(c, r, 0, -0.3, head_width=0.2, head_length=0.2, fc="k", ec="k")
            elif a == config.ACTIONS['DOWN']:
                ax.arrow(c, r, 0, 0.3, head_width=0.2, head_length=0.2, fc="k", ec="k")
            elif a == config.ACTIONS['LEFT']:
                ax.arrow(c, r, -0.3, 0, head_width=0.2, head_length=0.2, fc="k", ec="k")
            elif a == config.ACTIONS['RIGHT']:
                ax.arrow(c, r, 0.3, 0, head_width=0.2, head_length=0.2, fc="k", ec="k")

    ax.set_title(f"Policy after {episode_num} episodes")

    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)

    return im

def plot_final_policy(q_table, env, rewards, filename=None):
    """
    Plots the final learned policy and Q-values.
    """
    Q_grid = q_table.reshape((env.height, env.width, -1))

    best_actions = np.argmax(Q_grid, axis=2)
    best_values = np.max(Q_grid, axis=2)

    fig, ax = plt.subplots(figsize=(env.width * 0.8, env.height * 0.8))
    im = plot_policy_frame(ax, best_values, best_actions, env, rewards)
    plt.colorbar(im, ax=ax, label="Max Q-value")

    # plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def make_policy_gif(history_values, history_actions, env, rewards, filename="q_learning_policy.gif", fps=5):
    """
    Creates a GIF of Q-table evolution.
    """

    fig_width = env.width * 0.8
    fig_height = env.height * 0.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Initialize the plot with the first frame
    im = plot_policy_frame(ax, history_values[0], history_actions[0], env, rewards, episode_num=0)
    cbar = plt.colorbar(im, ax=ax, label="Max Q-value")

    def update(frame):
        ax.clear()
        episode_multiplier = 10
        if frame == len(history_values) - 1:
             episode_multiplier = (config.AGENT_PARAMS['num_episodes'] - 1) // (len(history_values) - 1) if len(history_values) > 1 else 1
             current_episode_num = (len(history_values) - 1) * episode_multiplier
        else:
             current_episode_num = frame * episode_multiplier # Adjust if sampling rate changes

        plot_policy_frame(ax, history_values[frame], history_actions[frame], env, rewards, episode_num=current_episode_num)

        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(history_values), blit=False, interval=1000/fps) # interval in ms

    # ani.save(filename, writer="pillow", fps=fps)
    plt.show()

    plt.close(fig) # Close the figure to free memory after saving