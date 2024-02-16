import unittest
import numpy as np
from Agents.Feedback_Agent_Stepwise import FeedbackAgentStepwise

env_names = ['CliffWalking-v0', "FrozenLake-v1"]
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4

# change index 0 = cliffwalk, 1 = frozenlake
chosen_env = 0

env_name = env_names[chosen_env]
learning_rate = 0.1
discount = 0.9
epsilon = 1  # 1 if epsilon should decay over time as well
epsilon_decay = 0.01
num_reward_updates = 1000
num_q_updates = 5
max_steps_per_episode = 50
noise = 0  # just for testing
delay = 5
batches = 5


class TestFeedbackAgentStepwise(unittest.TestCase):
    def setUp(self):
        self.feedback = {(0, 0): 1, (1, 1): 1, (5, 3): 1, (12, 2): 1}

        self.agent = FeedbackAgentStepwise(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                           max_steps_per_episode,
                                           num_reward_updates, num_q_updates, noise, delay, batches, feedback_variants[0])
        pass

    def test_get_feedback_IBF_correct(self):
        s, a = 5, 3
        feedback = self.agent._get_feedback_IBF(s, a)
        self.assertEqual(feedback, 1)

    def test_get_feedback_IBF_incorrect(self):
        s, a = 5, 1
        feedback = self.agent._get_feedback_IBF(s, a)
        self.assertEqual(feedback, 0)

    def test_get_feedback_ABF_correct(self):
        # last of tuple does not matter here
        trajectory = [(0, 0, 0), (1, 1, 0), (5, 3, 0)]
        feedback = self.agent._get_feedback_ABF(trajectory)
        self.assertEqual(feedback, 1)

    def test_get_feedback_ABF_incorrect(self):
        # last of tuple does not matter here
        trajectory = [(0, 1, 0), (1, 0, 0)]
        feedback = self.agent._get_feedback_ABF(trajectory)
        self.assertEqual(feedback, 0)

    def test_get_feedback_ABF_back_on_track(self):
        # exactly 0.8 correct
        # last of tuple does not matter here
        trajectory = [(20, 1, 0), (0, 0, 0), (1, 1, 0), (5, 3, 0), (12, 2, 0)]
        feedback = self.agent._get_feedback_ABF(trajectory)
        self.assertEqual(feedback, 0)

    def test_get_feedback_ABF_multiple_correct(self):
        # one optimal state action pair is more often than once
        # last of tuple does not matter here
        trajectory = [(0, 0, 0),(0, 0, 0), (1, 1, 0), (5, 3, 0)]
        feedback = self.agent._get_feedback_ABF(trajectory)
        self.assertEqual(feedback, 1)

    def test_get_feedback_SRF_correct(self):
        # last of tuple does not matter here
        trajectory = [(0, 0, 0), (1, 1, 0), (5, 3, 0)]
        feedback = self.agent._get_feedback_SRF(trajectory)
        self.assertEqual(feedback, 1)

    def test_get_feedback_SRF_incorrect(self):
        # last of tuple does not matter here
        trajectory = [(20, 1, 0)]
        feedback = self.agent._get_feedback_SRF(trajectory)
        self.assertEqual(feedback, 0)

    def test_get_feedback_SRF_back_on_track(self):
        # exactly 0.8 correct
        # last of tuple does not matter here
        trajectory = [(20, 1, 0), (0, 0, 0), (1, 1, 0), (5, 3, 0), (12, 2, 0)]
        feedback = self.agent._get_feedback_SRF(trajectory)
        self.assertEqual(feedback, 0.8)

    def test_epsilon_decay_IBF(self):
        num_episodes = 50
        agent = FeedbackAgentStepwise(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                      max_steps_per_episode,
                                      num_episodes, num_q_updates, 0.1, delay, batches, feedback_variants[0])
        agent.feedback_learning()
        test_epsilon = agent.epsilon

        calc_epsilon = 1
        min_epsilon = 0.01
        max_epsilon = 1
        for episode in range(num_episodes):
            calc_epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -epsilon_decay * episode)
        self.assertEqual(test_epsilon, calc_epsilon)

    def test_epsilon_decay_var4(self):
        num_episodes = 50
        agent = FeedbackAgentStepwise(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                      max_steps_per_episode,
                                      num_episodes, num_q_updates, 0.1, delay, batches,  feedback_variants[1])
        agent.feedback_learning()
        test_epsilon = agent.epsilon

        calc_epsilon = 1
        min_epsilon = 0.01
        max_epsilon = 1
        for episode in range(num_episodes):
            calc_epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -epsilon_decay * episode)
        self.assertEqual(test_epsilon, calc_epsilon)

    def test_epsilon_decay_var4(self):
        num_episodes = 50
        agent = FeedbackAgentStepwise(env_name, self.feedback, learning_rate, discount, epsilon, epsilon_decay,
                                      max_steps_per_episode,
                                      num_episodes, num_q_updates, 0.1, delay, batches,  feedback_variants[3])
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
