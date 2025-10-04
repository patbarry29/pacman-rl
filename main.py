import os
import pygame
import numpy as np
import sys
from matplotlib import pyplot as plt

import config

# Import modules from your project structure
from visualisation.pygame_renderer import PygameRenderer
from environment.grid_loader import load_grid
from environment.grid_env import GridEnv
from agents.q_learning_agent import QLearningAgent
from visualisation.q_table_visualizer import plot_final_policy, make_policy_gif
from utils.evaluation import evaluate_policy
from agents.monte_carlo_tree_search import TreeNode, monte_carlo_search, update_root_after_move

def run_human_game(env, renderer):
    """
    Allows a human player to control Pacman in the environment.
    """
    player_state = env.reset()
    game_over = False

    renderer.init_display("RL vs Pacman (Human Player)")

    while not game_over:
        current_display_grid = env.get_grid_snapshot()
        renderer.render(current_display_grid, player_pos=player_state)

        action, quit_game = renderer.handle_input()

        if quit_game:
            game_over = True
            break

        if action is not None:
            _, reward, done = env.step(action)
            player_state = env.player_pos

            if done:
                game_over = True
                print(f"Game Over! {'Win!' if reward == config.REWARDS['GOAL'] else 'Loss!'}")
                pygame.time.wait(2000)

    renderer.close_display()

def run_agent_visualization(env, agent, renderer, num_games_to_play, delay_ms):
    """
    Visualizes an agent playing the game on the Pygame grid for a few games.
    The agent uses its current Q-table (typically in exploitation mode).
    """
    renderer.init_display("RL vs Pacman (Agent Playing)")
    original_epsilon = agent.epsilon
    agent.epsilon = config.AGENT_PARAMS['min_epsilon']

    print(f"\nVisualizing agent playing {num_games_to_play} games...")
    should_quit_visualization = False

    for game_num in range(num_games_to_play):
        if should_quit_visualization:
            break
        player_pos = env.reset()
        game_over = False
        steps_this_game = 0
        print(f"Starting visualization game {game_num + 1}/{num_games_to_play}")

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    should_quit_visualization = True
                    print("Visualization interrupted by user (Pygame window close).")
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    game_over = True
                    should_quit_visualization = True
                    print("Visualization interrupted by user (ESC key).")
                    break
            if game_over:
                break
            current_display_grid = env.get_grid_snapshot()
            renderer.render(current_display_grid, player_pos=player_pos)
            action = agent.choose_action(player_pos)
            player_pos, reward, done = env.step(action)
            steps_this_game += 1
            if done:
                game_over=True
                game_state = 'win' if reward == config.REWARDS['GOAL'] else 'loss'
                result = 'Win!' if game_state == 'win' else 'Loss!'
                print(f"Game {game_num + 1} finished in {steps_this_game} steps. Result: {result}")
                renderer.render(env.get_grid_snapshot(), player_pos=player_pos)
                pygame.time.wait(1000)
            pygame.time.wait(delay_ms)


    pygame.time.wait(1000)
    renderer.close_display()
    agent.epsilon = original_epsilon


def execute_single_agent_training_mode(env):
    """
    Trains a single Q-Learning agent using parameters from config.py
    and plots its learning performance, also generates a policy GIF.
    """
    q_agent = QLearningAgent(env,
                            alpha=config.AGENT_PARAMS['alpha'],
                            epsilon=config.AGENT_PARAMS['epsilon'],
                            gamma=config.AGENT_PARAMS['gamma'],
                            decay=config.AGENT_PARAMS['decay'],
                            min_epsilon=config.AGENT_PARAMS['min_epsilon'])

    print("\nStarting Q-Learning training for a single agent...")
    final_q_table, history_values, history_actions = q_agent.train(
        num_episodes=config.AGENT_PARAMS['num_episodes'],
        max_steps_per_episode=config.AGENT_PARAMS['max_steps_per_episode']
    )

    q_agent.save_q_table(config.AGENT_PARAMS['q_table_save_path'])
    print(f"Trained Q-table saved to: {config.AGENT_PARAMS['q_table_save_path']}")

    print("\nGenerating policy GIF...")
    make_policy_gif(history_values, history_actions, env, config.REWARDS,
                    filename="policy_evolution_single_agent.gif")
    print("Policy GIF generated as policy_evolution_single_agent.gif")


def execute_agent_visualization_mode(env, renderer):
    """
    Handles the visualization of a pre-trained Q-Learning agent.
    """
    q_agent = QLearningAgent(env)
    q_table_to_load = config.BEST_Q_TABLE_PATH

    q_agent.load_q_table(q_table_to_load)

    run_agent_visualization(
        env,
        q_agent,
        renderer,
        num_games_to_play=1,
        delay_ms=config.AGENT_RENDER_DELAY_MS
    )

def execute_policy_plotting_mode(env):
    """
    Handles plotting the final policy from a Q-table.
    """
    q_agent = QLearningAgent(env)
    q_table_to_load = config.BEST_Q_TABLE_PATH

    try:
        q_agent.load_q_table(q_table_to_load)
        final_q_table = q_agent.q_table
        print(f"Loaded Q-table from: {q_table_to_load} for policy plotting.")
    except FileNotFoundError:
        print(f"Q-table not found at {q_table_to_load}. Performing a quick training run to generate a policy for plotting...")
        q_agent_for_plot_train = QLearningAgent(env,
                                                alpha=config.AGENT_PARAMS['alpha'],
                                                epsilon=config.AGENT_PARAMS['epsilon'],
                                                gamma=config.AGENT_PARAMS['gamma'],
                                                decay=config.AGENT_PARAMS['decay'],
                                                min_epsilon=config.AGENT_PARAMS['min_epsilon'])
        final_q_table, _, _ = q_agent_for_plot_train.train(
            num_episodes=config.AGENT_PARAMS.get('num_episodes_for_quick_train', 500)
        )
        print("Quick training complete.")

    print("\nGenerating final policy plot...")
    policy_plot_filename = os.path.join("experiment_results", "final_q_policy.png")
    plot_final_policy(final_q_table, env, config.REWARDS, filename=policy_plot_filename)
    print(f"Final policy plot generated and saved to {policy_plot_filename}")

# NEW FUNCTION FOR POLICY EVALUATION
def execute_policy_evaluation_mode(env, renderer):
    """
    Loads a trained agent and evaluates its policy, printing quantitative metrics.
    """
    q_agent = QLearningAgent(env)
    q_table_to_load = config.BEST_Q_TABLE_PATH

    q_agent.load_q_table(q_table_to_load)

    evaluate_policy(
        env,
        q_agent,
        num_eval_episodes=config.EVAL_PARAMS['num_eval_episodes'],
        max_steps_per_episode=config.EVAL_PARAMS['max_steps_per_episode'],
    )

def display_menu_and_get_choice():
    """ Displays the main menu and gets user input. """
    print("\n--- RL vs Pacman ---")
    print("Select mode:")
    print("1. Play as Human")
    print("2. Train Single Q-Learning Agent & Generate Policy GIF")
    print("3. Visualize Trained Q-Learning Agent (interactive game play)")
    print("4. Plot Final Q-Learning Policy (static image)")
    print("5. Evaluate Learned Policy (quantitative metrics)")
    return input("Enter choice (1-5): ")

def main():
    """ Main function to initialize components and run the selected mode. """
    # grid_filepath = config.ENV_PARAMS.get('grid_filepath', "grid.npy")
    grid_filepath = "grid.npy"
    grid, start_pos, goal_pos, ghost_positions = load_grid(grid_filepath)
    env = GridEnv(grid, start_pos, goal_pos, ghost_positions)
    renderer = PygameRenderer(env.height, env.width)

    choice = display_menu_and_get_choice()

    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    if choice == '1':
        run_human_game(env, renderer)
    elif choice == '2':
        execute_single_agent_training_mode(env)
    elif choice == '3':
        execute_agent_visualization_mode(env, renderer)
    elif choice == '4':
        execute_policy_plotting_mode(env)
    elif choice == '5':
        execute_policy_evaluation_mode(env, renderer)
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")

    if pygame.display.get_init():
        pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()










    grid_filepath = config.ENV_PARAMS.get('grid_filepath', "grid.npy")
    grid, start_pos, goal_pos, ghost_positions = load_grid(grid_filepath)
    env = GridEnv(grid, start_pos, goal_pos, ghost_positions)
    player_pos = env.reset()
    root = TreeNode(player_pos)

    done = False
    total_reward = 0
    i = 0

    while not done:
        action = monte_carlo_search(env, root, iterations=500, C=1.41)
        next_state, reward, done = env.step(action)
        total_reward += reward
        # Re-root the tree at the chosen child
        root = update_root_after_move(root, action, next_state)
        print(f"Action {i}: moved", next(key for key, value in config.ACTIONS.items() if value == action))

        i += 1

        if reward == config.REWARDS['GOAL']:
            print(f'\nVICTORY! in {i} steps\n')
        elif reward == config.REWARDS['GHOST']:
            print(f'\nDEFEAT! in {i} steps\n')

    print("Game finished with total reward:", total_reward)

    # Average Performance
    losses = 0
    i = 0
    wins = 0
    while i < 100:
        player_pos = env.reset()
        root = TreeNode(player_pos)
        done = False

        while not done:
            action = monte_carlo_search(env, root, iterations=1000, C=2)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # Re-root the tree at the chosen child
            root = update_root_after_move(root, action, next_state)

            i += 1

            if reward == config.REWARDS['GOAL']:
                wins += 1
            if reward == config.REWARDS['GHOST']:
                losses += 1

    print(wins/100)
    print(losses/100)