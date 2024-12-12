import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dss.metric import cov_metrics

# Define the Adult dataset environment
class AdultDatasetEnv(gym.Env):
    def __init__(self, limit: float = 0.2):
        super(AdultDatasetEnv, self).__init__()

        # Load and preprocess the Adult dataset
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 
                'capital_loss', 'hours_per_week', 'native_country', 'income']
        data.columns = columns
            # Creates binary labels

        categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 
                                            'relationship', 'race', 'sex', 'native_country']  # specify your categorical columns
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        data['income'] = data['income'].map({' <=50K': 0, ' >50K': 1,
                                            ' <=50K.': 0, ' >50K.': 1})
        # One-hot encode categorical columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns).astype(float)

        data_encoded[numeric_columns] = (data_encoded[numeric_columns] - data_encoded[numeric_columns].mean()) / data_encoded[numeric_columns].std()

        # Encode categorical variables
        #data = pd.get_dummies(data, columns=['workclass', 'education', 'marital_status', 'occupation', 
                                            #'relationship', 'race', 'sex', 'native_country'])
        self.features = data_encoded.drop(columns=['income']).values  # Drop target if not needed
        self.labels = data_encoded['income'].values  # Example binary labels


        # Create a DataLoader
        self.train_loader = DataLoader(AdultDataset(self.features, self.labels), batch_size=1, shuffle=True)

        # Initialize environment variables
        self.feature_dim = self.features.shape[1]
        self.limit = int(limit * len(self.train_loader))  # Limit number of samples to select

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(2)  # Actions: Include or Exclude
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)

        # Track steps and subset
        self.steps = 0
        self.samples = []
        self.labels = []
        self.current_diversity = 0  # Initial diversity metric value

    def reset(self,  cov_metric="trace",seed: int = 42):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []
        self.labels = []
        self.current_diversity = 1e-5
        self.steps = 0



        # Get the first data point as the initial observation
        first_sample, label = next(self.data_iter)

        self.metric = cov_metrics[cov_metric](self.feature_dim)  # Diversity metric initialization
        self.data_buffer = (first_sample, label)
        return first_sample.numpy(), {}

    def step(self, action: int):
        if action == 1:  # Include this data point
            # Retrieve current sample from data buffer
            sample, label = self.data_buffer
            self.samples.append(sample)
            self.labels.append(label)

            # Update diversity metric
            self.metric.update(sample.numpy())
            new_diversity = self.metric.get_diversity()

            # Calculate reward based on diversity change
            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0  # No diversity gain if sample is excluded

        self.steps += 1
        done = self.steps >= self.limit

        # Get the next sample or end the episode if there are no more samples
        try:
            next_sample, next_label = next(self.data_iter)
            self.data_buffer = (next_sample, next_label)
        except StopIteration:
            done = True
            self.data_buffer = (torch.zeros(self.feature_dim, dtype=torch.float32), torch.tensor(0))

        return self.data_buffer[0].numpy(), reward, done, {}, {}

# Define a custom Dataset for Adult data compatible with DataLoader
class AdultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Binary labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


