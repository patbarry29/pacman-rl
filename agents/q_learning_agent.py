import numpy as np
import config

class QLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9, decay=0.005, min_epsilon=0.05):
        self.env = env
        self.alpha = alpha     # Learning rate
        self.epsilon = epsilon # Exploration-exploitation trade-off (current value)
        self.gamma = gamma     # Discount factor
        self.decay = decay     # Epsilon decay rate per episode
        self.min_epsilon = min_epsilon # Minimum epsilon value during decay

        # Q-table: (num_states, num_actions)
        self.q_table = np.zeros((self.env.height * self.env.width, len(config.ACTIONS)))

        self.history_values = []
        self.history_actions = []
        self.history_steps = []
        self.history_reward = []

    def _state_to_idx(self, pos):
        # Converts (row, col) position to a linear state index.
        return pos[0] * self.env.width + pos[1]

    def choose_action(self, current_pos):
        state_idx = self._state_to_idx(current_pos)

        if np.random.rand() < self.epsilon:
            # explore
            return np.random.choice(len(config.ACTIONS))
        else:
            # exploit
            return np.argmax(self.q_table[state_idx])

    def learn(self, current_pos, action, reward, next_pos):
        state_idx = self._state_to_idx(current_pos)
        next_state_idx = self._state_to_idx(next_pos)

        old_value = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[next_state_idx]) # Max Q-value for the next state

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_idx, action] = new_value

    def train(self, num_episodes=500, max_steps_per_episode=200):
        min_steps_to_goal = np.inf

        self.history_actions = []
        self.history_values = []

        best_actions = np.argmax(self.q_table, axis=1).reshape(self.env.height, self.env.width)
        max_values   = np.max(self.q_table, axis=1).reshape(self.env.height, self.env.width)
        self.history_actions.append(best_actions.copy())
        self.history_values.append(max_values.copy())

        for episode in range(num_episodes):
            player_pos = self.env.reset()
            game_over = False
            steps_this_episode = 0
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(player_pos)

                old_player_pos = player_pos

                player_pos, reward, game_over = self.env.step(action)

                total_reward += reward

                self.learn(old_player_pos, action, reward, player_pos)

                steps_this_episode += 1

                if game_over:
                    break

            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.decay
            else:
                self.epsilon = self.min_epsilon # Ensure epsilon doesn't go below min

            if reward == config.REWARDS['GOAL'] and steps_this_episode < min_steps_to_goal:
                min_steps_to_goal = steps_this_episode

            # Store policy and values periodically for GIF generation
            # if episode % 10 == 0 or episode == num_episodes - 1:
            best_actions = np.argmax(self.q_table, axis=1).reshape(self.env.height, self.env.width)
            max_values   = np.max(self.q_table, axis=1).reshape(self.env.height, self.env.width)

            self.history_actions.append(best_actions.copy())
            self.history_values.append(max_values.copy())
            self.history_steps.append(steps_this_episode)
            self.history_reward.append(total_reward)

            # if (episode + 1) % 50 == 0:
            #     print(f"Episode {episode + 1}/{num_episodes}: Epsilon={self.epsilon:.2f}, Min steps to goal={min_steps_to_goal}")

        print(f"Training finished after {num_episodes} episodes. Min steps to goal: {min_steps_to_goal}")

        return self.q_table, self.history_values, self.history_actions

    def load_q_table(self, filepath):
        self.q_table = np.load(filepath)

    def save_q_table(self, filepath):
        np.save(filepath, self.q_table)