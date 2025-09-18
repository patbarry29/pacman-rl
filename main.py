import pygame
import numpy as np
import config

# Import modules from your project structure
from utils.helpers import make_grid
from visualisation.pygame_renderer import PygameRenderer
from environment.grid_loader import load_grid
from environment.grid_env import GridEnv
from agents.q_learning_agent import QLearningAgent
from visualisation.q_table_visualizer import plot_final_policy, make_policy_gif

def run_human_game(env, renderer):
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
            player_state = env.player_pos # Update player_state from env after step

            if done:
                game_over = True
                print(f"Game Over! {'Win!' if reward == config.REWARDS['GOAL'] else 'Loss!'}")
                pygame.time.wait(2000) # Show final state for a moment

        # pygame.time.wait(config.HUMAN_RENDER_DELAY_MS)

    renderer.close_display()

def run_agent_game(env, agent, renderer, num_episodes=1, training_mode=False):
    renderer.init_display(f"RL vs Pacman (Agent {'Training' if training_mode else 'Playing'})")

    original_epsilon = agent.epsilon
    original_decay = agent.decay

    if not training_mode:
        agent.epsilon = config.AGENT_PARAMS['min_epsilon']
        agent.decay = 0

    for episode in range(num_episodes):
        player_pos = env.reset()
        game_over = False
        print(f"Starting Episode {episode + 1}/{num_episodes}")

        while not game_over:
            current_display_grid = env.get_grid_snapshot()
            renderer.render(current_display_grid, player_pos=player_pos)

            action = agent.choose_action(player_pos)

            old_player_pos = player_pos
            player_pos, reward, done = env.step(action)

            if training_mode:
                agent.learn(old_player_pos, action, reward, player_pos)
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon -= agent.decay
                else:
                    agent.epsilon = agent.min_epsilon

            if done:
                game_over = True
                print(f"Episode {episode + 1} finished. Result: {'Win!' if reward == config.REWARDS['GOAL'] else 'Loss!'}")
                pygame.time.wait(1000)
                break

            pygame.time.wait(config.AGENT_RENDER_DELAY_MS)

    renderer.close_display()

    agent.epsilon = original_epsilon
    agent.decay = original_decay


if __name__ == "__main__":
    grid_filepath = "grid.npy"

    grid, start_pos, goal_pos, ghost_positions = load_grid(grid_filepath)

    # Initialize environment
    env = GridEnv(grid, start_pos, goal_pos, ghost_positions)

    # Initialize renderer
    renderer = PygameRenderer(env.height, env.width)

    # --- Choose Mode ---
    print("\n--- RL vs Pacman ---")
    print("Select mode:")
    print("1. Play as Human")
    print("2. Train Q-Learning Agent & Generate Policy GIF")
    print("3. Visualize Trained Q-Learning Agent (interactive game play)")
    print("4. Plot Final Q-Learning Policy (static image)")
    print("5. Exit")

    choice = input("Enter choice (1-5): ")

    if choice == '1':
        run_human_game(env, renderer)
    elif choice == '2':
        # Initialize Q-learning agent with parameters from config
        q_agent = QLearningAgent(env,
                                alpha=config.AGENT_PARAMS['alpha'],
                                epsilon=config.AGENT_PARAMS['epsilon'],
                                gamma=config.AGENT_PARAMS['gamma'],
                                decay=config.AGENT_PARAMS['decay'],
                                min_epsilon=config.AGENT_PARAMS['min_epsilon'])

        print("\nStarting Q-Learning training...")
        final_q_table, history_values, history_actions = q_agent.train(
            num_episodes=config.AGENT_PARAMS['num_episodes'],
            max_steps_per_episode=config.AGENT_PARAMS['max_steps_per_episode']
        )

        # Save the trained Q-table
        q_agent.save_q_table(config.AGENT_PARAMS['q_table_save_path'])

        print("Training complete. Generating policy GIF...")
        make_policy_gif(history_values, history_actions, env, config.REWARDS)
        print("GIF generation complete.")

    elif choice == '3':
        # Visualize trained Q-Learning Agent playing interactively
        q_agent = QLearningAgent(env) # Initialize with default parameters

        # Attempt to load a pre-trained Q-table
        q_agent.load_q_table(config.AGENT_PARAMS['q_table_save_path'])

        # If no Q-table was loaded (e.g., file not found or first run), offer to train a bit
        if np.all(q_agent.q_table == 0):
            print("No saved Q-table found or loaded. Performing a quick training run for visualization...")
            # Use specific parameters for quick training for visualization
            q_agent_for_viz_train = QLearningAgent(env,
                                            alpha=config.AGENT_PARAMS['alpha'],
                                            epsilon=config.AGENT_PARAMS['epsilon'],
                                            gamma=config.AGENT_PARAMS['gamma'],
                                            decay=config.AGENT_PARAMS['decay'],
                                            min_epsilon=config.AGENT_PARAMS['min_epsilon'])
            q_agent_for_viz_train.train(num_episodes=config.AGENT_PARAMS['num_episodes_for_viz_play'])
            q_agent = q_agent_for_viz_train # Use this agent for playing

        print("\nVisualizing Q-Learning Agent gameplay...")
        # Run agent for a few episodes in exploitation mode (low epsilon handled in run_agent_game)
        run_agent_game(env, q_agent, renderer, num_episodes=5, training_mode=False)

    elif choice == '4':
        # Plot the final policy without GIF
        q_agent = QLearningAgent(env) # Initialize with default parameters

        # Attempt to load a pre-trained Q-table
        q_agent.load_q_table(config.AGENT_PARAMS['q_table_save_path'])

        # If no Q-table was loaded, perform a quick training run
        if np.all(q_agent.q_table == 0):
            print("No saved Q-table found or loaded. Performing a quick training run to generate a policy for plotting...")
            q_agent_for_plot_train = QLearningAgent(env,
                                                    alpha=config.AGENT_PARAMS['alpha'],
                                                    epsilon=config.AGENT_PARAMS['epsilon'],
                                                    gamma=config.AGENT_PARAMS['gamma'],
                                                    decay=config.AGENT_PARAMS['decay'],
                                                    min_epsilon=config.AGENT_PARAMS['min_epsilon'])
            final_q_table, _, _ = q_agent_for_plot_train.train(
                num_episodes=config.AGENT_PARAMS['num_episodes_for_final_plot']
            )
            q_agent = q_agent_for_plot_train # Use this agent for plotting
        else:
            final_q_table = q_agent.q_table

        print("\nGenerating final policy plot...")
        plot_final_policy(final_q_table, env, config.REWARDS, filename="final_q_policy.png")
        print("Final policy plot generated.")

    elif choice == '5':
        print("Exiting game.")
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")