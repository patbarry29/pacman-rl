import numpy as np
import config

def evaluate_policy(env, agent, num_eval_episodes=100, max_steps_per_episode=None):
    print(f"\n--- Starting Policy Evaluation for {num_eval_episodes} episodes ---")

    # Set agent to pure exploitation mode for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    if max_steps_per_episode is None:
        max_steps_per_episode = config.AGENT_PARAMS['max_steps_per_episode']

    # Data collection lists
    successful_episodes = 0
    total_rewards = []
    total_steps = []
    ghost_collisions = 0
    steps_before_collision = [] # For failed episodes due to ghost
    wall_bumps = [] # List of wall bumps per episode
    min_distances_to_ghost_per_episode = [] # List of average min distances to ghost for each episode

    for _ in range(num_eval_episodes):
        player_pos = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        episode_wall_bumps = 0
        min_distances_this_episode = [] # distances to closest ghost for current episode

        # Get initial ghost positions (assuming static for now, or updated by env.step)
        current_ghost_positions = env.ghost_positions # Assuming this is accessible

        while not done and episode_steps < max_steps_per_episode:
            old_player_pos = player_pos

            action = agent.choose_action(player_pos) # Agent chooses action based on policy

            player_pos, reward, done = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Check for wall bump
            if player_pos == old_player_pos and reward == config.REWARDS['STEP'] and not done:
                episode_wall_bumps += 1

            # Calculate distance to closest ghost
            if current_ghost_positions: # Only if there are ghosts
                distances = [abs(player_pos[0] - g[0]) + abs(player_pos[1] - g[1]) for g in current_ghost_positions]
                min_distances_this_episode.append(min(distances))

            if done:
                if reward == config.REWARDS['GOAL']:
                    successful_episodes += 1
                elif reward == config.REWARDS['GHOST']:
                    ghost_collisions += 1
                    steps_before_collision.append(episode_steps) # Record steps if hit ghost
                # Other 'done' conditions (e.g., max steps reached without goal/ghost) handled implicitly

        # Store episode-level data
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        wall_bumps.append(episode_wall_bumps)
        if min_distances_this_episode:
            min_distances_to_ghost_per_episode.append(np.mean(min_distances_this_episode))
        else:
            min_distances_to_ghost_per_episode.append(np.nan) # No ghosts, so NaN

    # Restore agent's original epsilon
    agent.epsilon = original_epsilon

    # Calculate metrics
    success_rate = successful_episodes / num_eval_episodes
    avg_total_reward = np.mean(total_rewards)
    std_total_reward = np.std(total_rewards)
    avg_total_steps = np.mean(total_steps)
    std_total_steps = np.std(total_steps)

    ghost_collision_rate = ghost_collisions / num_eval_episodes
    avg_steps_before_ghost_collision = np.mean(steps_before_collision) if steps_before_collision else 0
    std_steps_before_ghost_collision = np.std(steps_before_collision) if steps_before_collision else 0

    avg_wall_bumps_per_episode = np.mean(wall_bumps)
    std_wall_bumps_per_episode = np.std(wall_bumps)

    # Filter out NaNs for average min distance if some episodes had no ghosts
    valid_min_distances = [d for d in min_distances_to_ghost_per_episode if not np.isnan(d)]
    avg_min_distance_to_ghost = np.mean(valid_min_distances) if valid_min_distances else np.nan
    std_min_distance_to_ghost = np.std(valid_min_distances) if valid_min_distances else np.nan

    metrics = {
        "num_eval_episodes": num_eval_episodes,
        "success_rate": success_rate,
        "avg_cumulative_reward": avg_total_reward,
        "std_cumulative_reward": std_total_reward,
        "avg_steps_per_episode": avg_total_steps,
        "std_steps_per_episode": std_total_steps,
        # "avg_steps_successful": avg_steps_successful,
        # "std_steps_successful": std_steps_successful,
        # "avg_steps_failed": avg_steps_failed,
        # "std_steps_failed": std_steps_failed,
        "ghost_collision_rate": ghost_collision_rate,
        "avg_steps_before_ghost_collision": avg_steps_before_ghost_collision,
        "std_steps_before_ghost_collision": std_steps_before_ghost_collision,
        "avg_wall_bumps_per_episode": avg_wall_bumps_per_episode,
        "std_wall_bumps_per_episode": std_wall_bumps_per_episode,
        "avg_min_distance_to_ghost_per_step": avg_min_distance_to_ghost, # New, clearer name
        "std_min_distance_to_ghost_per_step": std_min_distance_to_ghost,
    }

    print("\n--- Policy Evaluation Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k.replace('_', ' ').capitalize()}: {v:.4f}")
        else:
            print(f"- {k.replace('_', ' ').capitalize()}: {v}")
    print("---------------------------------\n")

    return metrics