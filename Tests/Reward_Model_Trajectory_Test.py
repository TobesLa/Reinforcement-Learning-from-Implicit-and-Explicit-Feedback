import unittest
import gymnasium as gym
from Reward_Models.Reward_Model_Trajectory import RewardModelTrajectory
import torch

env_name = 'CliffWalking-v0'
learning_rate = 0.5
discount = 0.99
epsilon = 0.2
num_episodes = 10000
max_steps_per_episode = 100
noise = 0.01
delay = 0.1
start_state = 36
optimal_val = -13
batches = 5


class TestRewardModelTrajectory(unittest.TestCase):
    def setUp(self):
        self.env = gym.make(env_name)
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        self.reward_model = RewardModelTrajectory(num_states, num_actions, batches, learning_rate, noise)

    def test_first_trajectory(self):
        test_trajectory_optimal = [[(start_state, 0)]]
        normal = self.reward_model._update_normal_dist(test_trajectory_optimal)
        self.assertEqual(normal.mean.item(), 1)

    def test_forward_single_sample(self):
        normal = torch.distributions.normal.Normal(0.5, noise)
        feedback_tensor = torch.tensor(0.5, dtype=torch.float, requires_grad=True)
        sum = self.reward_model.forward(normal, feedback_tensor)
        self.assertEqual(sum, normal.log_prob(feedback_tensor).sum().item())

    def test_forward_multiple_samples(self):
        normals = [torch.distributions.normal.Normal(0.5, noise) for _ in range(5)]
        feedback_tensors = [torch.tensor(0.5, dtype=torch.float, requires_grad=True) for _ in range(5)]
        log_prob_sum = 0.0
        for i in range(5):
            normal = normals[i]
            feedback_tensor = feedback_tensors[i]
            log_prob_sum += self.reward_model.forward(normal, feedback_tensor)
        manual_sum = sum(normal.log_prob(feedback).sum().item() for feedback, normal in zip(feedback_tensors, normals))
        self.assertEqual(log_prob_sum, manual_sum)


if __name__ == '__main__':
    unittest.main()
