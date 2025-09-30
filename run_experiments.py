import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Import modules from your project structure
import config
from environment.grid_loader import load_grid
from environment.grid_env import GridEnv
from agents.q_learning_agent import QLearningAgent

def run_hyperparameter_experiment():
    """
    Orchestrates the training and evaluation of multiple Q-Learning agents
    with different hyperparameter combinations, saving results and plots.
    This script is intended to be run independently.
    """
    # Initialize environment (needs to be done here as this is a standalone script)
    grid_filepath = config.ENV_PARAMS.get('grid_filepath', "grid.npy")
    grid, start_pos, goal_pos, ghost_positions = load_grid(grid_filepath)
    env = GridEnv(grid, start_pos, goal_pos, ghost_positions)

    # Setup output directories
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    q_tables_dir = os.path.join(output_dir, "q_tables")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(q_tables_dir, exist_ok=True)

    # Get experiment parameters from config
    experiment_params = config.EXPERIMENT_PARAMS
    alphas_to_test = experiment_params['alphas_to_test']
    gammas_to_test = experiment_params['gammas_to_test']
    decays_to_test = experiment_params['decays_to_test']
    num_runs_per_param = experiment_params['num_runs_per_param']

    # Agent parameters for training, mostly from config.AGENT_PARAMS
    num_episodes = config.AGENT_PARAMS['num_episodes']
    max_steps_per_episode = config.AGENT_PARAMS['max_steps_per_episode']
    min_epsilon = config.AGENT_PARAMS['min_epsilon']
    starting_epsilon = config.AGENT_PARAMS['epsilon'] # Use starting epsilon from config

    # Generate all hyperparameter combinations
    param_combinations = list(itertools.product(alphas_to_test, gammas_to_test, decays_to_test))

    all_experiment_data = [] # Stores aggregated results for summary plots

    print(f"\n--- Starting Hyperparameter Experimentation with {len(param_combinations)} combinations ---")

    # Initialize variables to track min/max values across all experiments for fixed plot scaling
    all_max_reward = float('-inf')
    all_min_reward = float('inf')
    all_max_steps = float('-inf')
    all_min_steps = float('inf')

    # First pass: Collect all training data and determine global min/max for plotting
    for i, (alpha, gamma, decay) in enumerate(param_combinations):
        print(f"\n--- Running Experiment {i+1}/{len(param_combinations)} ---")

        # Create a copy of base agent parameters and override for this run
        current_agent_params = config.AGENT_PARAMS.copy()
        current_agent_params['alpha'] = alpha
        current_agent_params['gamma'] = gamma
        current_agent_params['decay'] = decay

        # Generate unique identifier for this parameter set
        param_id = (f"alpha_{current_agent_params['alpha']:.2f}_"
                    f"gamma_{current_agent_params['gamma']:.2f}_"
                    f"epsilon_start_{starting_epsilon:.2f}_"
                    f"decay_{current_agent_params['decay']:.4f}").replace('.', 'p')

        # Initialize lists to store results from multiple runs for this param combination
        all_rewards_for_param_set = []
        all_steps_for_param_set = []
        all_q_tables_for_param_set = []

        print(f"  Parameters: alpha={alpha:.2f}, gamma={gamma:.2f}, decay={decay:.4f}")

        # Run multiple training sessions for each parameter combination
        for run_idx in range(num_runs_per_param):
            print(f"    Training Run {run_idx+1}/{num_runs_per_param}...")

            # Create a new agent for each run to ensure fresh start
            q_agent = QLearningAgent(env,
                                    alpha=current_agent_params['alpha'],
                                    epsilon=current_agent_params['epsilon'], # This is starting epsilon
                                    gamma=current_agent_params['gamma'],
                                    decay=current_agent_params['decay'],
                                    min_epsilon=min_epsilon)

            final_q_table, _, _ = q_agent.train( # We don't need history_values, history_actions for averaging
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode
            )

            all_rewards_for_param_set.append(q_agent.history_reward)
            all_steps_for_param_set.append(q_agent.history_steps)
            all_q_tables_for_param_set.append(final_q_table)

        # Calculate averages and standard deviations across runs for this parameter set
        avg_rewards = np.mean(all_rewards_for_param_set, axis=0)
        std_rewards = np.std(all_rewards_for_param_set, axis=0)
        avg_steps = np.mean(all_steps_for_param_set, axis=0)
        std_steps = np.std(all_steps_for_param_set, axis=0)
        avg_q_table = np.mean(all_q_tables_for_param_set, axis=0)

        # Save the averaged Q-table
        run_q_table_save_path = os.path.join(q_tables_dir, f"q_table_{param_id}_avg{num_runs_per_param}runs.npy")
        np.save(run_q_table_save_path, avg_q_table)
        print(f"  Averaged Q-table saved to: {run_q_table_save_path}")

        # Store aggregated results for later analysis and plotting
        all_experiment_data.append({
            'params': current_agent_params,
            'reward_history_avg': avg_rewards,
            'reward_history_std': std_rewards,
            'steps_history_avg': avg_steps,
            'steps_history_std': std_steps,
            'param_id': param_id,
        })

        # Update global min/max values for later plot scaling
        all_max_reward = max(all_max_reward, np.max(avg_rewards))
        all_min_reward = min(all_min_reward, np.min(avg_rewards))
        all_max_steps = max(all_max_steps, np.max(avg_steps))
        all_min_steps = min(all_min_steps, np.min(avg_steps))

    # Add some padding to the ranges for better plot visualization
    reward_range_padding = (all_max_reward - all_min_reward) * 0.1
    steps_range_padding = (all_max_steps - all_min_steps) * 0.1

    # Set fixed y-axis limits for all plots using the global min/max
    reward_y_min = all_min_reward - reward_range_padding
    reward_y_max = all_max_reward + reward_range_padding
    steps_y_min = all_min_steps - steps_range_padding
    steps_y_max = all_max_steps + steps_range_padding

    # Second pass: Generate individual plots with fixed axis ranges
    print("\n--- Generating individual plots for each experiment ---")
    for data in all_experiment_data:
        param_id = data['param_id']
        avg_rewards = data['reward_history_avg']
        std_rewards = data['reward_history_std']
        avg_steps = data['steps_history_avg']
        std_steps = data['steps_history_std']
        current_agent_params = data['params'] # Not strictly needed for plotting, but good context

        # Reward Plot with error bands and fixed y-axis
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(num_episodes), avg_rewards, 'b-', label='Mean reward')
        plt.fill_between(np.arange(num_episodes),
                         avg_rewards - std_rewards,
                         avg_rewards + std_rewards,
                         alpha=0.3, color='b', label='±1 std dev')
        plt.title(f"Average Cumulative Reward ({num_runs_per_param} runs)\n({param_id.replace('_', ' ').replace('p', '.')})")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.ylim(reward_y_min, reward_y_max)
        plt.grid(True)
        plt.legend()
        reward_plot_filename = os.path.join(plot_dir, f"reward_per_episode_{param_id}_avg{num_runs_per_param}runs.png")
        plt.savefig(reward_plot_filename)
        plt.close()
        print(f"  Saved: {reward_plot_filename}")

        # Steps Plot with error bands and fixed y-axis
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(num_episodes), avg_steps, 'g-', label='Mean steps')
        plt.fill_between(np.arange(num_episodes),
                         avg_steps - std_steps,
                         avg_steps + std_steps,
                         alpha=0.3, color='g', label='±1 std dev')
        plt.title(f"Average Steps per Episode ({num_runs_per_param} runs)\n({param_id.replace('_', ' ').replace('p', '.')})")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.ylim(steps_y_min, steps_y_max)
        plt.grid(True)
        plt.legend()
        steps_plot_filename = os.path.join(plot_dir, f"steps_per_episode_{param_id}_avg{num_runs_per_param}runs.png")
        plt.savefig(steps_plot_filename)
        plt.close()
        print(f"  Saved: {steps_plot_filename}")

    # --- Optional: Generate a summary plot comparing different runs (all on one graph) ---
    print("\n--- Generating summary comparison plots ---")
    plt.figure(figsize=(12, 8))
    for data in all_experiment_data:
        label = data['param_id'].replace('_', ' ').replace('p', '.')
        plt.plot(np.arange(num_episodes), data['reward_history_avg'], label=label)
    plt.title(f"Average Cumulative Reward per Episode for All Experiments ({num_runs_per_param} runs each)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.ylim(reward_y_min, reward_y_max)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    summary_reward_filename = os.path.join(plot_dir, f"summary_all_rewards_avg{num_runs_per_param}runs.png")
    plt.savefig(summary_reward_filename)
    plt.close()
    print(f"  Saved: {summary_reward_filename}")

    plt.figure(figsize=(12, 8))
    for data in all_experiment_data:
        label = data['param_id'].replace('_', ' ').replace('p', '.')
        plt.plot(np.arange(num_episodes), data['steps_history_avg'], label=label)
    plt.title(f"Average Steps per Episode for All Experiments ({num_runs_per_param} runs each)")
    plt.xlabel("Episode")
    plt.ylabel("Total Steps")
    plt.ylim(steps_y_min, steps_y_max)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    summary_steps_filename = os.path.join(plot_dir, f"summary_all_steps_avg{num_runs_per_param}runs.png")
    plt.savefig(summary_steps_filename)
    plt.close()
    print(f"  Saved: {summary_steps_filename}")

    print("\n--- All hyperparameter experiments completed! ---")

if __name__ == "__main__":
    run_hyperparameter_experiment()