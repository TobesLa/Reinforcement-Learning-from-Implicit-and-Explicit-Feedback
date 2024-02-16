import torch
import torch.nn as nn
import torch.optim as optim


class RewardModelStepwise(nn.Module):
    """
    PyTorch-based reward model for stepwise feedback in reinforcement learning.

    Attributes:

    - learning_rate (float): Learning rate for updating the reward model.

    - rewards (nn.Parameter): Learnable parameter representing the reward values for each state-action pair.

    - optimizer (torch.optim.Adam): Optimizer for updating the reward model parameters.

    - batches (int): Number of batches for updating the reward model estimate.
    """

    def __init__(self, num_states, num_actions, batches, learning_rate):
        super(RewardModelStepwise, self).__init__()
        self.rewards = nn.Parameter(torch.zeros((num_states, num_actions)))
        self.optimizer = optim.Adam([self.rewards], lr=learning_rate)
        self.batches = batches

    def forward(self, states, actions):
        """
        Forward pass to predict rewards for given states and actions.
        :param states: list [int] - list of states
        :param actions: list [int] - list of actions
        :return: torch - predicted reward using sigmoid
        """
        return torch.sigmoid(self.rewards[states, actions])

    def update_reward_model(self, states, actions, feedbacks):
        """
        Update the reward model based on provided states, actions, and feedbacks
        :param states: list [int] - list of states
        :param actions: list [int] - list of actions
        :param feedbacks: list [float] - list of feedbacks
        """
        self.optimizer.zero_grad()
        predicted_feedbacks = self.forward(states, actions)
        feedbacks_tensor = torch.tensor(feedbacks)
        loss = torch.abs(feedbacks_tensor - predicted_feedbacks)
        loss = loss.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def get_reward(self, state, action):
        """
        Get the predicted reward for a specific state-action pair.
        :param state: int
        :param action: int
        :return: float - predicted reward
        """
        return self.forward(state, action).item()
