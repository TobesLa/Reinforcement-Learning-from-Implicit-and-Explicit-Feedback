import unittest
import numpy as np
from Agents.Feedback_Agent_Trajectory import FeedbackAgentTrajectory

env_names = ['CliffWalking-v0', "FrozenLake-v1"]
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
# change index 0 = cliffwalk, 1 = frozenlake
chosen_env = 0

env_name = env_names[chosen_env]
learning_rate = 0.1
discount = 0.5
epsilon = 1  # 1 if epsilon should decay over time as well
epsilon_decay = 0.01
num_reward_updates = 1000
num_q_updates = 5
max_steps_per_episode = 50
noise = 0  # just for testing
delay = 5
batches = 5

class TestFeedbackAgentTrajectory(unittest.TestCase):
    def setUp(self):

        self.feedback = {(0, 0): 1, (1, 1): 1, (5, 3): 1, (12, 2): 1}

        self.agent = FeedbackAgentTrajectory(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                           max_steps_per_episode,
                                           num_reward_updates, num_q_updates, noise, delay, batches, feedback_variants[2])
        pass

    def test_get_feedback_ERF_correct(self):
        # last of tuple does not matter here
        trajectory = [(0, 0, 0), (1, 1, 0), (5, 3, 0)]
        feedback = self.agent._get_feedback_var3(trajectory)
        self.assertEqual(feedback, 1)

    def test_get_feedback_ERF_incorrect(self):
        # last of tuple does not matter here
        trajectory = [(0, 1, 0), (1, 0, 0)]
        feedback = self.agent._get_feedback_var3(trajectory)
        self.assertEqual(feedback, 0)

    def test_get_feedback_ERF_back_on_track(self):
        # exactly 0.8 correct
        # last of tuple does not matter here
        trajectory = [(20, 1, 0), (0, 0, 0), (1, 1, 0), (5, 3, 0), (12, 2, 0)]
        print(len(trajectory))
        feedback = self.agent._get_feedback_var3(trajectory)
        self.assertEqual(feedback, 0.8)

    def test_epsilon_decay_ERF(self):
        num_episodes = 50
        agent = FeedbackAgentTrajectory(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                      max_steps_per_episode,
                                      num_episodes, num_q_updates, 0.1, delay, batches, feedback_variants[2])
        agent.feedback_learning()
        test_epsilon = agent.epsilon

        calc_epsilon = 1
        min_epsilon = 0.01
        max_epsilon = 1
        for episode in range(num_episodes):
            calc_epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -epsilon_decay * episode)
        self.assertEqual(test_epsilon, calc_epsilon)


if __name__ == '__main__':
    unittest.main()
