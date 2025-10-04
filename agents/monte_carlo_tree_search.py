import numpy as np
import copy

class TreeNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None  # lazy init
        self.action_from_parent = action_from_parent  # useful when re-rooting

    def uct_score(self, C=1.41):
        if self.visits == 0:
            return float("inf")
        avg = self.total_reward / self.visits
        explore = C * np.sqrt(np.log(self.parent.visits) / self.visits)
        return avg + explore

    def best_child(self, C=1.41):
        return max(self.children.values(), key=lambda c: c.uct_score(C))


class MCTree:
    def __init__(self, root):
        self.root = root

    def select(self, node):
        return max(node.children, key=lambda c: c.uct_score())

    def backpropagate(self, node, reward):
        # walk up the tree updating stats
        while node is not None:
            node.update(reward)
            node = node.parent


def rollout(env, state, max_depth=200):
    """Random rollout from state using simulate_step."""
    total_reward = 0
    for _ in range(max_depth):
        actions = env.get_valid_actions(state)
        if not actions:
            break
        action = np.random.choice(actions)
        state, reward, done = env.simulate_step(state, action)
        total_reward += reward
        if done:
            break
    return total_reward


def simulate(env, node, max_depth=120):
    """Play randomly from node.state until terminal or depth cutoff."""
    sim_env = copy.deepcopy(env)
    sim_env.player_pos = node.state
    total_reward = 0
    for _ in range(max_depth):
        action = np.random.choice(sim_env.action_space)
        _, reward, done = sim_env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def monte_carlo_search(env, root, iterations=500, C=1.41):
    """Perform MCTS from the given root node, return best action."""
    for _ in range(iterations):
        node = root
        state = node.state

        # 1. Selection: descend tree until a leaf or unexpanded action
        while node.untried_actions == [] and node.children:
            node = node.best_child(C)
            state = node.state

        # 2. Expansion: expand *one* untried action
        if node.untried_actions is None:
            node.untried_actions = env.get_valid_actions(state)

        if node.untried_actions:
            action = node.untried_actions.pop()
            next_state, reward, done = env.simulate_step(state, action)
            child = TreeNode(next_state, parent=node, action_from_parent=action)
            node.children[action] = child
            node = child
            state = next_state

        # 3. Simulation: heuristic-guided rollout
        reward = rollout(env, state)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # Best action from root = most visited child
    best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_action


def update_root_after_move(root, actual_action, new_state):
    """Reuse the subtree after taking a real action."""
    if actual_action in root.children:
        new_root = root.children[actual_action]
        new_root.parent = None
        return new_root
    else:
        return TreeNode(new_state)