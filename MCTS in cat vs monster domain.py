import random
import math
import matplotlib.pyplot as plt
import numpy as np

class State:
    def __init__(self, cat_pos, monster_pos, grid_size, turn):
        self.cat_pos = cat_pos
        self.monster_pos = monster_pos
        self.grid_size = grid_size
        self.turn = turn  # 'cat' or 'monster'

    def is_terminal(self):
        # Cat escapes if it reaches the edge of the grid
        if self.cat_pos[0] == 0 or self.cat_pos[0] == self.grid_size - 1 or \
           self.cat_pos[1] == 0 or self.cat_pos[1] == self.grid_size - 1:
            return True
        # Monster catches the cat
        if self.cat_pos == self.monster_pos:
            return True
        return False

    def get_winner(self):
        if self.cat_pos == self.monster_pos:
            return 'monster'
        if self.cat_pos[0] == 0 or self.cat_pos[0] == self.grid_size - 1 or \
           self.cat_pos[1] == 0 or self.cat_pos[1] == self.grid_size - 1:
            return 'cat'
        return None

    def get_legal_actions(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        pos = self.cat_pos if self.turn == 'cat' else self.monster_pos
        legal_actions = []
        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                legal_actions.append((dx, dy))
        return legal_actions

    def take_action(self, action):
        new_cat_pos = self.cat_pos
        new_monster_pos = self.monster_pos

        if self.turn == 'cat':
            new_cat_pos = (self.cat_pos[0] + action[0], self.cat_pos[1] + action[1])
        else:
            new_monster_pos = (self.monster_pos[0] + action[0], self.monster_pos[1] + action[1])

        return State(new_cat_pos, new_monster_pos, self.grid_size, 'monster' if self.turn == 'cat' else 'cat')

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        actions = self.state.get_legal_actions()
        for action in actions:
            if not any(child.state.cat_pos == self.state.take_action(action).cat_pos and \
                       child.state.monster_pos == self.state.take_action(action).monster_pos for child in self.children):
                new_state = self.state.take_action(action)
                child_node = Node(new_state, self)
                self.children.append(child_node)
                return child_node

    def update(self, result):
        self.visits += 1
        self.value += result


def mcts(initial_state, iterations):
    root = Node(initial_state)

    for _ in range(iterations):
        node = root
        # Selection
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not node.state.is_terminal():
            node = node.expand()

        # Simulation
        current_state = node.state
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            action = random.choice(actions)
            current_state = current_state.take_action(action)

        # Backpropagation
        result = 1 if current_state.get_winner() == 'cat' else -1
        while node is not None:
            node.update(result)
            result = -result
            node = node.parent

    return root.best_child(0).state

# Simulation tracking
class Agent:
    def __init__(self, grid_size, iterations):
        self.grid_size = grid_size
        self.iterations = iterations
        self.rewards = []

    def run_episode(self):
        initial_state = State(cat_pos=(self.grid_size // 2, self.grid_size // 2), monster_pos=(0, 0), grid_size=self.grid_size, turn='cat')
        total_reward = 0

        for _ in range(100):  # Limit the steps per episode
            if initial_state.is_terminal():
                winner = initial_state.get_winner()
                total_reward = 1 if winner == 'cat' else -1
                break

            best_state = mcts(initial_state, self.iterations)
            initial_state = best_state

        self.rewards.append(total_reward)

    def train(self, episodes):
        for _ in range(episodes):
            self.run_episode()

    def evaluate_policy(self, episodes):
        policy_rewards = []
        for _ in range(episodes):
            initial_state = State(cat_pos=(self.grid_size // 2, self.grid_size // 2), monster_pos=(0, 0), grid_size=self.grid_size, turn='cat')
            total_reward = 0

            for _ in range(100):  # Limit the steps per episode
                if initial_state.is_terminal():
                    winner = initial_state.get_winner()
                    total_reward = 1 if winner == 'cat' else -1
                    break

                best_state = mcts(initial_state, self.iterations)
                initial_state = best_state

            policy_rewards.append(total_reward)
        return policy_rewards

    def evaluate_learned_policy(self, episodes):
        win_rate = 0
        for _ in range(episodes):
            initial_state = State(cat_pos=(self.grid_size // 2, self.grid_size // 2), monster_pos=(0, 0), grid_size=self.grid_size, turn='cat')
            while not initial_state.is_terminal():
                best_state = mcts(initial_state, self.iterations)
                initial_state = best_state
            if initial_state.get_winner() == 'cat':
                win_rate += 1
        return win_rate / episodes

# Main execution
agent = Agent(grid_size=5, iterations=500)
agent.train(episodes=200)  # Increased episodes for more simulation

# Plot Learning Progress
plt.plot(agent.rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Learning Progress of MCTS algo in cat vs monster domain')
plt.show()

# Evaluate Learned Policy Performance
policy_rewards = agent.evaluate_policy(episodes=200)  # More evaluation episodes
average_reward = np.mean(policy_rewards)
print(f"Average Total Reward: {average_reward:.2f}")

win_rate = agent.evaluate_learned_policy(episodes=200)
print(f"Win Rate for Learned Policy: {win_rate * 100:.2f}%")
# Plot Policy Performance
plt.hist(policy_rewards, bins=10, alpha=0.7, label="Optimized Learned Policy")
plt.xlabel('Total Reward')
plt.ylabel('Frequency')
plt.title('Policy Performance for MCTS in cat vs monster domain')
plt.legend()
plt.show()