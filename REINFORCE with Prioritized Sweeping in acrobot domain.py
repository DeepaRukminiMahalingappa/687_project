import numpy as np
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Define the updated Acrobot environment
class AcrobotEnvironment:
    def __init__(self, state_size=6, target_height=5, reward_threshold=-100, max_steps=50):
        self.state_size = state_size  # Discretized state space
        self.target_height = target_height  # Target height for free end
        self.reward_threshold = reward_threshold  # Threshold for termination
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_steps = 0  # Step counter
        self.goal_state = target_height  # Designated goal height
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_steps = 0
        self.state = random.randint(0, self.state_size - 1)  # Random initial state
        return self.state

    def step(self, state, action):
        """Take a step in the environment."""
        next_state = (state + action) % self.state_size
        reward = -1  # Penalize every step
        
        # Check if target height is reached
        if next_state == self.target_height:
            reward = 0  # Reward for reaching the target height
            done = True
        else:
            done = False
        
        # Add a penalty for the number of steps taken
        reward -= 0.1 * self.current_steps  # Penalize more for using extra steps
        
        # Update the step count and check for termination condition
        self.current_steps += 1
        if self.current_steps >= self.max_steps or reward == 0:
            done = True
        return next_state, reward, done

# Prioritized Sweeping agent for learning
class PrioritizedSweeping:
    def __init__(self, env, gamma=0.99, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.q_table = np.zeros((env.state_size, env.state_size))  # Q-table with state-action values
        self.model = {}  # Model to store transitions and rewards

    def update_model(self, state, action, reward, next_state):
        self.model[(state, action)] = (reward, next_state)

    def plan(self, n=10):
        """Perform prioritized sweeping."""
        pq = PriorityQueue()
        for (state, action), (reward, next_state) in self.model.items():
            td_error = abs(reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])
            if td_error > self.theta:
                pq.put((-td_error, (state, action)))

        # Perform planning steps
        for _ in range(n):
            if pq.empty():
                break
            _, (state, action) = pq.get()
            reward, next_state = self.model[(state, action)]
            best_next_action = np.argmax(self.q_table[next_state])
            self.q_table[state, action] = reward + self.gamma * self.q_table[next_state, best_next_action]

            # Update priorities for predecessors
            for pred_state in range(self.env.state_size):
                for pred_action in range(self.env.state_size):
                    if (pred_state, pred_action) in self.model:
                        r, s = self.model[(pred_state, pred_action)]
                        td_error = abs(r + self.gamma * np.max(self.q_table[s]) - self.q_table[pred_state, pred_action])
                        if td_error > self.theta:
                            pq.put((-td_error, (pred_state, pred_action)))

    def learn(self, episodes=500, epsilon=1.0, epsilon_decay=0.99, n=10):
        state_size, action_size = self.env.state_size, self.env.state_size
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            epsilon = max(0.05, epsilon * epsilon_decay)
            total_reward = 0

            for _ in range(self.env.max_steps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, action_size)  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit

                next_state, reward, done = self.env.step(state, action)
                total_reward += reward

                # Update model
                self.update_model(state, action, reward, next_state)

                # Perform Q-learning update
                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[state, action] += 0.1 * (reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])

                # Prioritized Sweeping
                self.plan(n)

                state = next_state
                if done:
                    break

            rewards.append(total_reward)
            if episode % 50 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

        return rewards

# Main Experiment
state_size = 6
target_height = 5  # Example target height
env = AcrobotEnvironment(state_size, target_height)
agent = PrioritizedSweeping(env)

# Learn and Evaluate
rewards = agent.learn(episodes=500)

# Plot Results
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Learning Progress with Prioritized Sweeping Algorithm in acrobot domain')
plt.show()

# Policy Evaluation
def evaluate_policy(agent, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state = agent.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done = agent.env.step(state, action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return total_rewards

# Evaluate Learned Policy
policy_rewards = evaluate_policy(agent)
print(f"Average Total Reward: {np.mean(policy_rewards):.2f}")

# Plot Policy Performance
plt.hist(policy_rewards, bins=10, alpha=0.7, label="Learned Policy Performance of Prioritized Sweeping in acrobot domain")
plt.xlabel('Total Reward')
plt.ylabel('Frequency')
plt.title('Policy Performance ')
plt.legend()
plt.show()
