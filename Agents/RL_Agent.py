import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
import seaborn as sns
from matplotlib.colors import ListedColormap


class RLAgent:
    """
    A reinforcement learning agent based on Q-learning for simulating human feedback in the gymnasium environments
    'CliffWalking-v0' and 'FrozenLake-v1' and comparison to the feedback learning agents.

    Attributes:

    - env_name (str): Name of the environment, should be 'CliffWalking-v0' or 'FrozenLake-v1'.

    - learning_rate (float): Learning rate for the Q-learning algorithm.

    - discount (float): Discount factor for future rewards.

    - epsilon (float): Exploration-exploitation trade-off parameter.

    - num_episodes (int): Number of training episodes.

    - max_steps_per_episode (int): Maximum number of steps allowed per episode.

    """
    def __init__(self, env_name, learning_rate, discount, epsilon, num_episodes, max_steps_per_episode):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.env = gym.make(self.env_name)
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))

        self.rewards_all_episodes = []

        # if we want epsilon to decay
        self.exploration_decay_rate = 0.0005 if env_name == "FrozenLake-v1" else 0.01
        self.max_epsilon = epsilon
        self.min_epsilon = 0.01

    def q_learning(self):
        """
        Perform Q-learning to train the agent.
        :return:
            numpy array : the trained Q-table,
            list[float] : the rewards received for each episode
        """
        # explore decay from : https://www.youtube.com/watch?v=HGeI30uATws&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=9&ab_channel=deeplizard

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            curr_reward = 0

            for step in range(self.max_steps_per_episode):
                exploration_threshold = random.uniform(0, 1)
                if exploration_threshold < self.epsilon:
                    action = self.env.action_space.sample()  # explore
                else:
                    action = np.argmax(self.q_table[state, :])  # exploit

                next_state, reward, done, info, _ = self.env.step(action)

                # Update Q-value using the Q-learning update rule
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
                        reward + self.discount * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
                )

                curr_reward += reward
                state = next_state

                if done:
                    break
            # update epsilon to decay over time
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.exploration_decay_rate * episode)
            self.rewards_all_episodes.append(curr_reward)

        return self.q_table, self.rewards_all_episodes

    def save_policy(self, goal_state):
        """
        Save the learned policy as a plot.
        :param goal_state:(int, int) - Goal state coordinates.
        """
        optimal_policy = np.argmax(self.q_table, axis=1)

        # Visualize the policy using arrows and colors
        sns.set()
        if self.env_name == "FrozenLake-v1":
            shape = (4, 4)
            left = 0
            down = 1
            right = 2
            up = 3
            colors = ['black', 'orange', 'lightblue', 'lightgreen', '#FF0000']
        else:
            # cliffwalking
            shape = self.env.shape
            up = 0
            right = 1
            down = 2
            left = 3
            colors = ['black', '#FF0000', 'lightgreen', 'lightblue', 'orange']
        optimal_policy = optimal_policy.reshape(shape)
        print_policy = [['' for _ in range(optimal_policy.shape[1])] for _ in range(optimal_policy.shape[0])]
        for i in range(optimal_policy.shape[0]):
            for j in range(optimal_policy.shape[1]):
                curr_action = optimal_policy[i][j]
                if curr_action == up:
                    print_policy[i][j] = '↑'
                elif curr_action == right:
                    print_policy[i][j] = '→'
                elif curr_action == down:
                    print_policy[i][j] = '↓'
                elif curr_action == left:
                    print_policy[i][j] = '←'

        print_policy[goal_state[0]][goal_state[1]] = 'G'  # Goal state

        optimal_policy[goal_state[0]][goal_state[1]] = -1

        custom_cmap = ListedColormap(colors) # custom colormap so that for both envs, a direction has a specified color

        sns.heatmap(optimal_policy, annot=np.array(print_policy), fmt='', cmap=custom_cmap,
                    cbar=False, linewidths=.5, square=True)

        plt.title('Optimal Policy ')
        plt.savefig(self.env_name + 'policy.pdf')
        plt.close()

    def visualize_policy(self, q_table, env, goal_state):
        """
        Visualize the learned policy using arrows and colors.

        :param q_table: numpy array - The trained Q-table.

        :param env: gymnasium environment - The environment.

        :param goal_state:(int, int) - Goal state coordinates.
        """
        optimal_policy = np.argmax(q_table, axis=1)

        # Visualize the policy using arrows and colors
        sns.set()
        if self.env_name == "FrozenLake-v1":
            shape = (4, 4)
            left = 0
            down = 1
            right = 2
            up = 3
            colors = ['black', 'orange', 'lightblue', 'lightgreen', '#FF0000']
        else:
            # cliffwalking
            shape = env.shape
            up = 0
            right = 1
            down = 2
            left = 3
            colors = ['black', '#FF0000', 'lightgreen', 'lightblue', 'orange']
        optimal_policy = optimal_policy.reshape(shape)
        print_policy = [['' for _ in range(optimal_policy.shape[1])] for _ in range(optimal_policy.shape[0])]
        for i in range(optimal_policy.shape[0]):
            for j in range(optimal_policy.shape[1]):
                curr_action = optimal_policy[i][j]
                if curr_action == up:
                    print_policy[i][j] = '↑'
                elif curr_action == right:
                    print_policy[i][j] = '→'
                elif curr_action == down:
                    print_policy[i][j] = '↓'
                elif curr_action == left:
                    print_policy[i][j] = '←'

        print_policy[goal_state[0]][goal_state[1]] = 'G'  # Goal state

        optimal_policy[goal_state[0]][goal_state[1]] = -1

        custom_cmap = ListedColormap(colors) # custom colormap so that for both envs, a direction has a specified color

        sns.heatmap(optimal_policy, annot=np.array(print_policy), fmt='', cmap=custom_cmap,
                    cbar=False, linewidths=.5, square=True)
        plt.title(f'Optimal Policy')
        plt.show()

    def plot_rewards(self, optimal_val):
        """
        Plot the rewards earned during training episodes.
        :param optimal_val: int - representing optimal value
        """
        plt.plot(np.arange(self.num_episodes), self.rewards_all_episodes, label='Episode Rewards', color='red')
        plt.axhline(y=optimal_val, color='black', linestyle='--', label='Optimal Value')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(self.env_name + ' Rewards per Episode learned by Q-Learning')
        plt.legend()
        plt.show()

    def visualize_results(self):
        """
        Visualize the results of the trained agent's actions with the gymnasium's GUI.
        """
        env = gym.make(self.env_name, render_mode="human")  # to show gui on trained Agent
        # env = gym.make(self.env_name)
        state, _ = env.reset()
        total_reward = 0
        all_actions = []
        all_states = []
        done = False
        while not done:
            all_states.append(state)
            action = np.argmax(self.q_table[state, :])
            next_state, reward, done, info, _ = env.step(action)
            all_actions.append(action)
            total_reward += reward
            state = next_state

            if done:
                break
        print("**** Total reward ****", total_reward)
        print("**** all Actions ****", all_actions)
        print("**** all states ****", all_states)
        # 0 = UP, 1 = RIGHT,  2 = DOWN, 3 = LEFT

    def extract_feedback(self):
        """
        Extract the feedback dict from the agent's learned policy.
        :return:
            feedback: dict{(state, action):feedback} - extracted feedback,
            total_reward: float - total reward received during episode
        """
        env = gym.make(self.env_name)
        state, _ = env.reset()
        total_reward = 0
        feedback = {}
        done = False
        for step in range(100):
            action = np.argmax(self.q_table[state, :])
            next_state, reward, done, info, _ = env.step(action)
            feedback[(state, action)] = 1
            total_reward += reward
            state = next_state

            if done:
                break
        return feedback, total_reward

    def get_training_result(self):
        """
        Get the total accumulated reward of one episode from the agent exploiting its current policy.

        :return:
            float: Total accumulated reward.
        """
        state, _ = self.env.reset()
        accumulated_reward = 0
        for step in range(100):
            action = np.argmax(self.q_table[state, :])
            next_state, reward, done, _, _ = self.env.step(action)
            accumulated_reward += reward
            state = next_state
            if done:
                break
        return accumulated_reward
