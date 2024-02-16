import numpy as np
import gymnasium as gym
import random
from Reward_Models.Reward_Model_Trajectory import RewardModelTrajectory
from queue import Queue


class FeedbackAgentTrajectory:
    """
     A reinforcement learning agent that learns from simulated feedback by continuously updating a reward model
     based on trajectory-level feedback. The reward model is updated once per episode using the entire trajectory
     of the current episode.

     Attributes:

     - env_name (str): Name of the environment, should be 'CliffWalking-v0' or 'FrozenLake-v1'.

     - feedback (dict): A dictionary of {(state, action): feedback} where feedback is typically 1
       and used for training the approximate reward model.

     - learning_rate (float): Learning rate for updating the Q-table and the reward model (default is 0.1).

     - discount (float): Discount factor for future rewards (default is 0.5).

     - epsilon (float): Exploration-exploitation trade-off parameter (default is 1).

     - exploration_decay_rate (float): Rate of exploration decay over time (default is 0.01).

     - max_steps_per_episode (int): Maximum number of steps allowed per episode (default is 100).

     - num_episodes (int): Number of training episodes (default is 1000).

     - num_q_updates (int): Number of Q-value updates per episode (default is 5).

     - noise (float): The simulated noise of the feedback provided (default is 0.1).

     - delay (int): Number of time steps before applying feedback (default is 5).

     - batches (int): Number of batches for updating the reward model estimate (default is 5).

     - feedback_variant (str): Variant of feedback, should be 'ERF' (default is 'ERF') where
       ERF is Episodic Ratio Feedback.

     """

    def __init__(self, env_name, feedback, learning_rate=0.1, discount=0.5, epsilon=1, exploration_decay_rate=0.01,
                 max_steps_per_episode=100,
                 num_episodes=1000, num_q_updates=5,
                 noise=0.1, delay=5, batches=5, feedback_variant='ERF'):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.num_q_updates = num_q_updates
        self.noise = noise
        self.delay = delay
        self.feedback_variant = feedback_variant
        assert feedback_variant == 'ERF', 'Feedback variant not allowed'
        self.feedback = feedback
        self.env = gym.make(self.env_name)
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.batches = batches
        self.reward_model = RewardModelTrajectory(self.num_states, self.num_actions, self.batches, learning_rate, noise)
        self.true_rewards_all_episodes = []
        self.rm_rewards_all_episodes = []

        self.exploration_decay_rate = exploration_decay_rate
        self.max_epsilon = epsilon
        self.min_epsilon = 0.01

    def feedback_learning(self):
        """
        Perform feedback-based learning using trajectory-level feedback.

        :return:
            numpy array : the trained Q-table,
            list[float] : the true rewards received for each episode,
            list[float] : the estimated rewards received for each episode
        """
        delay_batch = Queue()
        batched_trajectories = {}  # episode:trajectory
        batched_normalized_trajectories = []
        batched_feedbacks = []
        rm_curr_reward = 0
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            true_curr_reward = 0

            curr_trajectory = []

            for step in range(self.max_steps_per_episode):
                action = self.__choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                curr_trajectory.append((state, action, next_state))
                true_curr_reward += reward
                state = next_state
                if done:
                    break

            self.true_rewards_all_episodes.append(true_curr_reward)
            curr_feedback = self.__get_feedback(curr_trajectory)
            normalized_traj = [(state, action) for state, action, _ in curr_trajectory]
            batched_trajectories[episode] = curr_trajectory
            batched_feedbacks.append(curr_feedback)
            batched_normalized_trajectories.append(normalized_traj)

            if episode == self.num_episodes - 1 or episode != 0 and episode % self.batches == 0:
                # put in queue all the batched data
                delay_batch.put(
                    (batched_trajectories,
                     batched_feedbacks,
                     batched_normalized_trajectories
                     )
                )
                # Reset batched data for the next round
                batched_trajectories = {}
                batched_feedbacks = []
                batched_normalized_trajectories = []

            # check if delay passed
            if not delay_batch.empty():
                top_batch = delay_batch.queue[0]
                last_episode, last_trajectory = list(top_batch[0].items())[-1]
                while not delay_batch.empty() and last_episode + self.delay == episode:
                    # Update Reward Model
                    top_batch = delay_batch.get()
                    delayed_batched_traj = top_batch[0]
                    delayed_feedbacks = top_batch[1]
                    delayed_normalized_traj = top_batch[2]

                    self.reward_model.update_reward_model(delayed_normalized_traj, delayed_feedbacks)

                    # Update Q-table
                    for curr_episode, trajectory in delayed_batched_traj.items():
                        for state, action, next_state in trajectory:
                            rm_reward = self.reward_model.get_reward(state, action)
                            rm_curr_reward += rm_reward
                            for _ in range(self.num_q_updates):
                                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * \
                                                              (rm_reward + self.discount * np.max(
                                                                  self.q_table[next_state, :]) -
                                                               self.q_table[state, action])
                        self.rm_rewards_all_episodes.append(rm_curr_reward)
                        rm_curr_reward = 0
                        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                            -self.exploration_decay_rate * curr_episode)

        # update remaining elements
        while not delay_batch.empty():
            top_batch = delay_batch.get()
            # print('top batch: ', top_batch)
            delayed_batched_traj = top_batch[0]
            delayed_feedbacks = top_batch[1]
            delayed_normalized_traj = top_batch[2]

            self.reward_model.update_reward_model(delayed_normalized_traj, delayed_feedbacks)

            # Update Q-table
            for curr_episode, trajectory in delayed_batched_traj.items():
                for state, action, next_state in trajectory:
                    rm_reward = self.reward_model.get_reward(state, action)
                    rm_curr_reward += rm_reward
                    for _ in range(self.num_q_updates):
                        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * \
                                                      (rm_reward + self.discount * np.max(
                                                          self.q_table[next_state, :]) -
                                                       self.q_table[state, action])
                self.rm_rewards_all_episodes.append(rm_curr_reward)
                rm_curr_reward = 0
                # update epsilon to decay over time
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                    -self.exploration_decay_rate * curr_episode)

        return self.q_table, self.true_rewards_all_episodes, self.rm_rewards_all_episodes

    def __choose_action(self, state):
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold < self.epsilon:
            # explore
            action = self.env.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.q_table[state, :])
        return action

    def __get_feedback(self, trajectory):
        if self.feedback_variant == 'ERF':
            return self._get_feedback_var3(trajectory)
        else:
            assert False, f'got unknown feedback variant: {self.feedback_variant}'

    def _get_feedback_var3(self, trajectory):
        feedback_list = [*self.feedback]
        num_optimal_steps = 0
        for state, action, _ in trajectory:
            if (state, action) in feedback_list:
                num_optimal_steps += 1
        feedback = num_optimal_steps / len(trajectory)
        return self.__create_noisy_feedback(feedback)

    def __create_noisy_feedback(self, feedback):
        """
        Add simulated noise to the feedback value.
        :param feedback: float - in range (0,1) representing simulated human feedback
        :return: float : feedback influenced by noise
        """
        lower_bound = max(0, feedback - self.noise)
        upper_bound = min(1, feedback + self.noise)
        noisy_f = random.uniform(lower_bound, upper_bound)
        return noisy_f
