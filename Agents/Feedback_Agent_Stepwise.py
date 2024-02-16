import numpy as np
import gymnasium as gym
import random
from Reward_Models.Reward_Model_Stepwise import RewardModelStepwise
from Reward_Models.Reward_Model_Trajectory import RewardModelTrajectory
from queue import Queue


class FeedbackAgentStepwise:
    """
    A reinforcement learning agent that learns from simulated feedback by continuously updating a reward model
    on each step during training. The reward model can be based on different feedback variants.

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

    - feedback_variant (str): Variant of feedback, should be 'IBF', 'ABF', or 'SRF' (default is 'IBF')
      where IBF is Individual Binary Feedback ABF is Aggregated Binary Feedback and SRF is Step-wise Ratio Feedback.
    """

    def __init__(self, env_name, feedback, learning_rate=0.1, discount=0.5, epsilon=1, exploration_decay_rate=0.01,
                 max_steps_per_episode=100,
                 num_episodes=1000, num_q_updates=5,
                 noise=0.1, delay=5, batches=5, feedback_variant='IBF'):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.num_q_updates = num_q_updates
        self.noise = noise
        self.delay = delay
        self.batches = batches
        self.feedback_variant = feedback_variant
        assert feedback_variant == 'IBF' or feedback_variant == 'ABF' or feedback_variant == 'SRF', 'Feedback ' \
                                                                                                       'variant not ' \
                                                                                                       'allowed '
        self.feedback = feedback
        self.env = gym.make(self.env_name)
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # create model from Feedback
        if feedback_variant == 'IBF':
            self.reward_model = RewardModelStepwise(self.num_states, self.num_actions, self.batches, self.learning_rate)
        else:
            self.reward_model = RewardModelTrajectory(self.num_states, self.num_actions, self.batches, learning_rate,
                                                      noise)
        self.true_rewards_all_episodes = []
        self.rm_rewards_all_episodes = []

        # we want epsilon to decay
        self.epsilon_decay_interval = num_episodes / 20
        self.exploration_decay_rate = exploration_decay_rate
        self.max_epsilon = epsilon
        self.min_epsilon = 0.01

        self.start_episode = 0

    def feedback_learning(self):
        """
        Perform feedback-based learning with stepwise reward updates.

        :return:
            numpy array : the trained Q-table,
            list[float] : the true rewards received for each episode,
            list[float] : the estimated rewards received for each episode
        """
        if self.feedback_variant == 'IBF':
            return self.__learning_feedback_IBF()
        delay_batch = Queue()
        batched_trajectories = []
        episodes_batch = []
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
                curr_feedback = self.__get_feedback(state, action, curr_trajectory)
                true_curr_reward += reward

                normalized_traj = [(state, action) for state, action, _ in curr_trajectory]
                batched_trajectories.append(curr_trajectory)
                episodes_batch.append(episode)
                batched_feedbacks.append(curr_feedback)
                batched_normalized_trajectories.append(normalized_traj)
                state = next_state

                if step == self.max_steps_per_episode - 1 or (step != 0 and len(batched_feedbacks) % self.batches == 0) or done :
                    # put in queue all the batched data
                    delay_batch.put(
                        (batched_trajectories,
                         batched_feedbacks,
                         batched_normalized_trajectories,
                         episodes_batch
                         )
                    )
                    # Reset batched data for the next round
                    batched_trajectories = []
                    batched_feedbacks = []
                    batched_normalized_trajectories = []
                    episodes_batch = []

                if done:
                    break

            # check if delay passed
            if not delay_batch.empty():
                top_batch = delay_batch.queue[0]
                last_episode = top_batch[3][-1]

                while not delay_batch.empty() and last_episode + self.delay == episode:
                    last_episode = top_batch[3][-1]
                    top_batch = delay_batch.get()
                    delayed_episodes = top_batch[3]
                    delayed_normalized_trajs = top_batch[2]
                    delayed_feedbacks = top_batch[1]
                    delayed_trajectories = top_batch[0]

                    end = False
                    if self.delay == 0 and delay_batch.empty() and any(
                            epi == self.num_episodes - 1 for epi in delayed_episodes):
                        end = True

                    self.reward_model.update_reward_model(delayed_normalized_trajs, delayed_feedbacks)

                    self.__update_q_table_base(delayed_trajectories, delayed_episodes, rm_curr_reward, end)

            self.true_rewards_all_episodes.append(true_curr_reward)

        # update remaining elements
        while not delay_batch.empty():
            top_batch = delay_batch.get()
            delayed_batched_traj = top_batch[0]
            delayed_feedbacks = top_batch[1]
            delayed_normalized_traj = top_batch[2]
            delayed_episodes = top_batch[3]
            self.reward_model.update_reward_model(delayed_normalized_traj, delayed_feedbacks)
            end = False
            if delay_batch.empty() and any(epi == self.num_episodes - 1 for epi in delayed_episodes):
                end = True
            self.__update_q_table_base(delayed_batched_traj, delayed_episodes, rm_curr_reward, end)

        return self.q_table, self.true_rewards_all_episodes, self.rm_rewards_all_episodes

    def __update_q_table_base(self, delayed_batched_traj, delayed_episodes, rm_curr_reward, end=False):
        for curr_episode, trajectory in zip(delayed_episodes, delayed_batched_traj):
            if curr_episode > self.start_episode:
                self.rm_rewards_all_episodes.append(rm_curr_reward)
                rm_curr_reward = 0
                self.start_episode = curr_episode
                # update epsilon to decay over time
                self.__decay_epsilon(curr_episode)
            for state, action, next_state in trajectory:
                rm_reward = self.reward_model.get_reward(state, action)
                rm_curr_reward += rm_reward
                for _ in range(self.num_q_updates):
                    self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * \
                                                  (rm_reward + self.discount * np.max(
                                                      self.q_table[next_state, :]) -
                                                   self.q_table[state, action])
        if end:
            self.rm_rewards_all_episodes.append(rm_curr_reward)

    # since we don't batch trajectories but states and actions we created an own version just for IBF
    def __learning_feedback_IBF(self):
        """
        Perform feedback-based learning with stepwise reward updates but specifically for feedback_variant var1
        :return:
            numpy array : the trained Q-table,
            list[float] : the true rewards received for each episode,
            list[float] : the estimated rewards received for each episode
        """
        delay_batch = Queue()
        trajectory_batch = []
        episodes_batch = []
        batched_feedbacks = []
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            true_curr_reward = 0
            rm_curr_reward = 0
            curr_trajectory = []
            for step in range(self.max_steps_per_episode):
                action = self.__choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                true_curr_reward += reward
                curr_trajectory.append((state, action, next_state))
                trajectory_batch.append((state, action, next_state))
                episodes_batch.append(episode)
                curr_feedback = self.__get_feedback(state, action, curr_trajectory)
                state = next_state
                batched_feedbacks.append(curr_feedback)
                if step == self.max_steps_per_episode - 1 or (step != 0 and len(batched_feedbacks) % self.batches == 0) or done :
                    # put in queue all the batched data
                    delay_batch.put(
                        (trajectory_batch,
                         batched_feedbacks,
                         episodes_batch
                         )
                    )
                    # Reset batched data for the next round
                    trajectory_batch = []
                    batched_feedbacks = []
                    episodes_batch = []

                if done:
                    break
            # check if delay passed
            if not delay_batch.empty():
                top_batch = delay_batch.queue[0]
                last_episode = top_batch[2][-1]
                while not delay_batch.empty() and last_episode + self.delay == episode:
                    last_episode = top_batch[2][-1]
                    top_batch = delay_batch.get()
                    delayed_batched_traj = top_batch[0]
                    states, actions, next_states = zip(*delayed_batched_traj)
                    delayed_feedbacks = top_batch[1]
                    delayed_episodes = top_batch[2]
                    end = False
                    if self.delay == 0 and delay_batch.empty() and any(
                            epi == self.num_episodes - 1 for epi in delayed_episodes):
                        end = True

                    self.reward_model.update_reward_model(states, actions, delayed_feedbacks)

                    self.__update_q_table_IBF(delayed_batched_traj, delayed_episodes, rm_curr_reward, end)

            self.true_rewards_all_episodes.append(true_curr_reward)
        # update remaining elements
        while not delay_batch.empty():
            top_batch = delay_batch.get()
            delayed_batched_traj = top_batch[0]
            states, actions, next_states = zip(*delayed_batched_traj)
            delayed_feedbacks = top_batch[1]
            delayed_episodes = top_batch[2]

            self.reward_model.update_reward_model(states, actions, delayed_feedbacks)
            end = False
            if delay_batch.empty() and any(epi == self.num_episodes - 1 for epi in delayed_episodes):
                end = True

            self.__update_q_table_IBF(delayed_batched_traj, delayed_episodes, rm_curr_reward, end)

        return self.q_table, self.true_rewards_all_episodes, self.rm_rewards_all_episodes

    def __update_q_table_IBF(self, delayed_batched_traj, delayed_episodes, rm_curr_reward, end=False):
        for (state, action, next_state), curr_episode in zip(delayed_batched_traj, delayed_episodes):
            if curr_episode > self.start_episode:
                self.rm_rewards_all_episodes.append(rm_curr_reward)
                rm_curr_reward = 0
                self.start_episode = curr_episode
                # update epsilon to decay over time
                self.__decay_epsilon(curr_episode)

            rm_reward = self.reward_model.get_reward(state, action)
            rm_curr_reward += rm_reward
            for _ in range(self.num_q_updates):
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * \
                                              (rm_reward + self.discount * np.max(
                                                  self.q_table[next_state, :]) -
                                               self.q_table[state, action])
        if end:
            self.rm_rewards_all_episodes.append(rm_curr_reward)

    def __decay_epsilon(self, curr_episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.exploration_decay_rate * curr_episode)

    def __choose_action(self, state):
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold < self.epsilon:
            # explore
            action = self.env.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.q_table[state, :])
        return action

    def __get_feedback(self, state, action, trajectory):
        if self.feedback_variant == 'IBF':
            return self._get_feedback_IBF(state, action)
        elif self.feedback_variant == 'ABF':
            return self._get_feedback_ABF(trajectory)
        elif self.feedback_variant == 'SRF':
            return self._get_feedback_SRF(trajectory)
        else:
            assert False, f'got unknown feedback variant: {self.feedback_variant}'

    def _get_feedback_IBF(self, state, action):
        curr_feedback = self.feedback.get((state, action), 0)
        return self.__create_noisy_feedback(curr_feedback)

    def _get_feedback_ABF(self, trajectory):
        # now returns 1 even if an optimal state action pair is more often used than once
        for state, action, _ in trajectory:
            if (state, action) not in self.feedback or self.feedback[(state, action)] != 1:
                return self.__create_noisy_feedback(0)
        return self.__create_noisy_feedback(1)

    def _get_feedback_SRF(self, trajectory):
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
