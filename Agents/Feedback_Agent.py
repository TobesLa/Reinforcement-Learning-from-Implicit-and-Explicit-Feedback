from matplotlib.colors import ListedColormap

from Agents.Feedback_Agent_Stepwise import FeedbackAgentStepwise
from Agents.Feedback_Agent_Trajectory import FeedbackAgentTrajectory
import numpy as np
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt


class FeedbackAgent:
    """
    A wrapper class that selects an appropriate feedback learning agent implementation based on the feedback variant.

    Attributes:

    - env_name (str): Name of the environment, should be 'CliffWalking-v0' or 'FrozenLake-v1'.

    - feedback (dict): A dictionary of {(state, action): feedback} where feedback is typically 1
      and represents the optimal state action.

    - learning_rate (float): Learning rate for the agent (default is 0.1).

    - discount (float): Discount factor for future rewards (default is 0.5).

    - epsilon (float): Exploration-exploitation trade-off parameter (default is 1).

    - exploration_decay_rate (float): Rate of exploration decay over time (default is 0.01).

    - max_steps_per_episode (int): Maximum number of steps allowed per episode (default is 100).

    - num_episodes (int): Number of training episodes (default is 1000).

    - num_q_updates (int): Number of Q-value updates per episode (default is 5).

    - noise (float): The simulated noise of the feedback provided (default is 0.1).

    - delay (int): Number of time steps before applying feedback (default is 5).

    - batches (int): Number of batches for updating reward model estimate  (default is 5).

    - feedback_variant (str): Variant of feedback, choose from 'IBF', 'ABF', 'ERF', 'SRF' (default is 'IBF')
      where IBF is Individual Binary Feedback ABF is Aggregated Binary Feedback ERF is Episodic Ratio Feedback
      and SRF is Step-wise Ratio Feedback.
    """

    def __init__(self, env_name, feedback, learning_rate=0.1, discount=0.5, epsilon=1, exploration_decay_rate=0.01,
                 max_steps_per_episode=100,
                 num_episodes=1000, num_q_updates=5,
                 noise=0.1, delay=5, batches=5, feedback_variant='IBF'):

        self.feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF']
        self.env_names = ['CliffWalking-v0', "FrozenLake-v1"]
        self.__check_parameters(env_name, batches, delay, discount, epsilon, exploration_decay_rate, feedback_variant,
                                learning_rate, max_steps_per_episode, noise, num_episodes, num_q_updates)

        if feedback_variant == 'IBF' or feedback_variant == 'ABF' or feedback_variant == 'SRF':
            self.agent = FeedbackAgentStepwise(env_name, feedback, learning_rate, discount, epsilon,
                                               exploration_decay_rate,
                                               max_steps_per_episode,
                                               num_episodes, num_q_updates, noise, delay, batches,
                                               feedback_variant)
        elif feedback_variant == 'ERF':
            self.agent = FeedbackAgentTrajectory(env_name, feedback, learning_rate, discount, epsilon,
                                                 exploration_decay_rate,
                                                 max_steps_per_episode,
                                                 num_episodes, num_q_updates, noise, delay, batches,
                                                 feedback_variant)
        else:
            assert False, 'incorrect Feedback Variant'

    def __check_parameters(self, env_name, batches, delay, discount, epsilon, exploration_decay_rate, feedback_variant,
                           learning_rate, max_steps_per_episode, noise, num_episodes, num_q_updates):
        if feedback_variant not in self.feedback_variants:
            raise ValueError("This Feedback variant is not allowed!")
        if learning_rate < 0 or learning_rate > 1:
            raise ValueError("Learning rate has to be in range [0,1]!")
        if discount < 0 or discount > 1:
            raise ValueError("Discount has to be in range [0,1]!")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon has to be in range [0,1]!")
        if exploration_decay_rate < 0 or exploration_decay_rate > 1:
            raise ValueError("Exploration_decay_rate has to be in range [0,1]!")
        if max_steps_per_episode < 0 or not isinstance(max_steps_per_episode, int):
            raise ValueError("Max_steps_per_episode has to be a positive int!")
        if num_episodes < 0 or not isinstance(num_episodes, int):
            raise ValueError("Number of episodes has to be a positive int!")
        if num_q_updates < 0 or not isinstance(num_q_updates, int):
            raise ValueError("Number of Q-updates has to be a positive int!")
        if noise < 0 or noise > 1:
            raise ValueError("Noise has to be in range [0,1]!")
        if delay < 0 or not isinstance(delay, int):
            raise ValueError("Delay has to be a positive int!")
        if batches < 1 or not isinstance(batches, int):
            raise ValueError("Batches has to be a positive int!")
        if env_name not in self.env_names:
            raise ValueError("Environment has to be cliffwalking or frozenlake!")

    def feedback_learning(self):
        """
        Perform feedback-based learning using the corresponding agent to the feedback variant.

        :return:
            numpy array : the trained Q-table,
            list[float] : the true rewards received for each episode,
            list[float] : the estimated rewards received for each episode
        """
        return self.agent.feedback_learning()

    def get_training_result(self):
        """
        Get the total accumulated reward of one episode from the agent exploiting its current policy.

        :return:
            float: Total accumulated reward.
        """
        state, _ = self.agent.env.reset()
        accumulated_reward = 0
        for step in range(self.agent.max_steps_per_episode):
            action = np.argmax(self.agent.q_table[state, :])
            next_state, reward, done, _, _ = self.agent.env.step(action)
            accumulated_reward += reward
            state = next_state
            if done:
                break
        return accumulated_reward

    def visualize_results(self):
        """
        Visualize the results of the trained agent's actions with the gymnasium's GUI.
        """
        env = gym.make(self.agent.env_name, render_mode="human")  # to show gui on trained Agent
        state, _ = env.reset()
        total_reward = 0
        all_actions = []
        done = False
        for step in range(self.agent.max_steps_per_episode):
            action = np.argmax(self.agent.q_table[state, :])
            all_actions.append(action)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        print("**** Total reward ****", total_reward)
        print("**** all Actions ****", all_actions)

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
        if self.agent.env_name == "FrozenLake-v1":
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

        custom_cmap = ListedColormap(colors)  # custom colormap so that for both envs, a direction has a specified color

        sns.heatmap(optimal_policy, annot=np.array(print_policy), fmt='', cmap=custom_cmap,
                    cbar=False, linewidths=.5, square=True)
        plt.title(f'Found Policy from Feedback {self.agent.feedback_variant}')
        plt.show()

    def plot_rewards(self, optimal_val):
        """
        Plot the rewards earned during training episodes.

        :param optimal_val: int - The optimal value for comparison.
        """
        plt.plot(np.arange(self.agent.num_episodes), self.agent.true_rewards_all_episodes, label='True Episode Rewards',
                 color='blue', zorder=3)
        # comment out to see the reward approximations during training
        #plt.plot(np.arange(self.agent.num_episodes), self.agent.rm_rewards_all_episodes,
                 #label='Episode Rewards from Reward Model learned from feedback', color='red', zorder=2)
        # plt.ylim(-100, 100)  # might need to comment out or change values depending on problem!
        plt.axhline(y=optimal_val, color='black', linestyle='--', label='Optimal Value', zorder=1)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(
            self.agent.env_name + ' Rewards per Episode earned by learning from Feedback ' + self.agent.feedback_variant)
        plt.legend()
        plt.show()
