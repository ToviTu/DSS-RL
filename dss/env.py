import gymnasium as gym
import numpy as np
import math
import torch
from torchvision import datasets, transforms

from .metric import cov_metrics
from .transformation import FlatteningFeature

class FashionMNISTEnv(gym.Env):
    def __init__(self, limit:float=.2):
        super(FashionMNISTEnv, self).__init__()

        # Initialize dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        self.feature_mapping = FlatteningFeature()

        # Initialize environment variables
        self.steps = 0
        self.samples = []
        self.labels = []
        self.current_diversity = 0# Arbitrary small number
        self.limit = int(limit * len(self.train_loader)) # Terminates when reaching the limit

        self.action_space = gym.spaces.Discrete(2) # In/Exclude
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)

    def reset(self, cov_metric="trace", seed: int=42):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []  # Clear sample buffer
        self.labels = []
        self.current_diversity = 1e-5

        # Return the first data point as the initial observation
        image, label = next(self.data_iter)

        feature_vec = self.feature_mapping(image)
        self.metric = cov_metrics[cov_metric](feature_vec.shape[1])

        self.data_buffer = (image, label)
        return image.squeeze().numpy(), {}

    def step(self, action:int):
        if action == 1:  # Include this data point
            image, label = self.data_buffer
            self.samples.append(image)
            self.labels.append(label)

            # Update diversity metric (online algorithm)
            self.metric.update(self.feature_mapping(image).squeeze().numpy())
            new_diversity = self.metric.get_diversity()

            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0

        self.steps += 1

        if len(self.samples) >= self.limit:
            done = True
            next_image, next_label = torch.zeros((1, 28, 28)), torch.zeros(1)
        else:
            done = False
            next_image, next_label = next(self.data_iter, (torch.zeros((1, 28, 28)), torch.zeros(1)))

        if torch.equal(next_image, torch.zeros((1, 28, 28))) and torch.equal(next_label, torch.zeros(1)):
            done = True  # No more data points

        self.data_buffer = (next_image, next_label)
        info = {"step": self.steps}  # Add basic info dictionary
        assert not math.isnan(reward)
        return next_image.squeeze().numpy(), reward, done, {}, {}