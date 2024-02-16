import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RewardModelTrajectory(nn.Module):
    """
        PyTorch-based reward model for feedback provided per trajectory in reinforcement learning.

        Attributes:

        - learning_rate (float): Learning rate for updating the reward model.

        - rewards (nn.Parameter): Learnable parameter representing the reward values for each state-action pair.

        - optimizer (torch.optim.Adam): Optimizer for updating the reward model parameters.

        - batches (int): Number of batches for updating the reward model estimate.
    """

    def __init__(self, num_states, num_actions, batches, learning_rate, noise):
        super(RewardModelTrajectory, self).__init__()
        self.rewards = nn.Parameter(torch.zeros((num_states, num_actions), requires_grad=True))
        self.optimizer = optim.Adam([self.rewards], lr=learning_rate)
        self.batches = batches
        self.noise = noise
        # std. dev. = 0 is not allowed for torch.normal therefore low default value
        if noise == 0:
            self.noise = 0.01

    def _update_normal_dist(self, trajectories):
        """
        Update of the normal distribution that is used to calculate the log probability
        of the feedback.
        :param trajectories: list(list((state,action))) - list of trajectories that is used to update reward model
        :return: torch: normal - a normal distribution with mean = sum of exponantiated rewards of all trajectories / size of trajectories
        """
        trajectory_states, trajectory_actions = zip(
            *[(state, action) for traj in trajectories for state, action in traj])
        trajectory_sum = torch.sum(torch.exp(self.rewards[trajectory_states, trajectory_actions]))

        len_all_trajectories = len(trajectory_states)

        mean = trajectory_sum / len_all_trajectories

        normal = torch.distributions.normal.Normal(mean, self.noise)
        return normal

    def get_reward(self, state, action):
        """
        Get the predicted reward for a specific state-action pair.
        :param state: int
        :param action: int
        :return: float - predicted reward
        """
        return self.rewards[state][action].item()

    def forward(self, normal, feedback_tensor):
        """
        Forward pass to calculate the log probability of the feedback.
        :param normal: torch - normal distribution
        :param feedback_tensor: torch - simulated feedbacks provided during training
        :return: log probability of the feedbacks
        """
        log_prob_sum = normal.log_prob(feedback_tensor)
        return log_prob_sum

    def update_reward_model(self, trajectories, feedbacks):
        """
        Update the reward model with the trajectories and corresponding feedback received during training.
        :param trajectories: list(list((state,action))) - list of trajectories that is used to update reward model
        :param feedbacks: list(int) - list of simulated feedbacks provided during training
        """
        self.optimizer.zero_grad()
        normal = self._update_normal_dist(trajectories)
        feedback_tensors = [torch.tensor(feedback, dtype=torch.float, requires_grad=True) for feedback in feedbacks]
        log_prob_sum = sum(self.forward(normal, feedback_tensor) for feedback_tensor in feedback_tensors)
        loss = -log_prob_sum
        loss.backward()
        self.optimizer.step()
