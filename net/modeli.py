import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(n_input_channels, 255),
            nn.ReLU(),
            nn.Linear(255, 255),
            nn.ReLU(),
            nn.Linear(255, features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)
