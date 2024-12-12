import gymnasium as gym
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tqdm
from os import path


from .metric import cov_metrics
from .transformation import *
from .utils import *


# Make new dataset with features
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, raw_inputs, targets, feature_mapping):
        self.raw_inputs = raw_inputs
        self.targets = targets

        # Precompute features
        if path.exists("playground/FMNIST_features.pt"):
            print("Loading precomputed features...")
            self.features = torch.load("playground/FMNIST_features.pt")
        else:
            print("Precomputing features...")
            loader = DataLoader(self.raw_inputs, batch_size=32, shuffle=False)
            features = []
            for image in tqdm.tqdm(loader):
                image = image if image.ndim == 4 else image.unsqueeze(1)
                feature_vec = feature_mapping(image)
                features.append(feature_vec)
            self.features = torch.cat(features, dim=0)
            torch.save(self.features, "playground/FMNIST_features.pt")
        
        assert len(self.raw_inputs) == len(self.features) == len(self.targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.raw_inputs[idx], self.features[idx], self.targets[idx]

class FeatureDataset_(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_mapping, name="features"):
        self.dataset = dataset

        # Precompute features
        if path.exists(f"playground/{name}.pt"):
            print("Loading precomputed features...")
            self.features = torch.load(f"playground/{name}.pt")
        else:
            print("Precomputing features...")
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            features = []
            for image, _ in tqdm.tqdm(loader):
                image = image if image.ndim == 4 else image.unsqueeze(1)
                feature_vec = feature_mapping(image)
                features.append(feature_vec)
            self.features = torch.cat(features, dim=0)
            torch.save(self.features, f"playground/{name}.pt")
        
        assert len(self.dataset) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.features[idx], self.dataset[idx][1]

class TextFeatureDataset_(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_mapping, name="features"):
        self.dataset = dataset

        # Precompute features
        if path.exists(f"playground/{name}.pt"):
            print("Loading precomputed features...")
            self.features = torch.load(f"playground/{name}.pt")
        else:
            print("Precomputing features...")
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            features = []
            for text in tqdm.tqdm(loader):
                feature_vec = feature_mapping(text['text'])
                features.append(feature_vec)
            self.features = torch.cat(features, dim=0)
            torch.save(self.features, f"playground/{name}.pt")
        
        assert len(self.dataset) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        features = self.features[idx]
        datum['feature'] = features

        return datum


class FashionMNISTEnv(gym.Env):
    def __init__(self, limit:float=.2, feature_mapping=FlatteningFeature()):
        super(FashionMNISTEnv, self).__init__()

        # Initialize dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        train_data = train_dataset.data.float() / 255.
        train_labels = train_dataset.targets

        self.dataset = FeatureDataset(train_data, train_labels, feature_mapping)
        self.train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            
        # Initialize environment variables
        self.samples = []
        self.labels = []
        self.current_diversity = 0 # Arbitrary small number
        self.limit = int(limit * len(self.train_loader)) # Terminates when reaching the limit

        self.action_space = gym.spaces.Discrete(2) # In/Exclude
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)

    def reset(self, cov_metric="trace", seed: int=42, use_raw=True):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []  # Clear sample buffer
        self.labels = []
        self.current_diversity = 1e-5
        self.use_raw = use_raw

        # Return the first data point as the initial observation
        image, feature, label = next(self.data_iter)
        feature_vec = image if use_raw else feature

        self.metric = cov_metrics[cov_metric](feature_vec.shape[1])

        self.data_buffer = (image, feature, label)
        return image.squeeze().numpy(), {}

    def step(self, action:int):
        if action == 1:  # Include this data point
            image, feature, label = self.data_buffer
            self.samples.append(image)
            self.labels.append(label)

            # Update diversity metric (online algorithm)
            feature_vec = feature if self.use_raw else feature
            
            self.metric.update(feature_vec.squeeze().numpy())
            new_diversity = self.metric.get_diversity()

            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0

        if len(self.samples) >= self.limit:
            next_image, next_feature, next_label = None, None, None
        else:
            next_image, next_feature, next_label = next(self.data_iter, (None, None, None))

        done = next_image is None # No more data points
             
        self.data_buffer = (next_image, next_feature, next_label)
        assert not math.isnan(reward)
        if next_image is not None: # Return the next observation
            next_image = next_image.squeeze().numpy()
        return next_image, reward, done, {}, {}


class CIFAR10Env(gym.Env):
    def __init__(
            self,
            limit:float=.2,
            cov_metric="trace", 
            feature_mapping=ResNetFeature(), 
            use_raw=True
        ):
        super(CIFAR10Env, self).__init__()
        self.use_raw = use_raw
        self.cov_metric = cov_metric

        # Initialize dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

        self.dataset = FeatureDataset_(train_dataset, feature_mapping, name="CIFAR10_features")
        self.train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            
        # Initialize environment variables
        self.samples = []
        self.labels = []
        self.current_diversity = 0 # Arbitrary small number
        self.limit = int(limit * len(self.train_loader)) # Terminates when reaching the limit

        self.action_space = gym.spaces.Discrete(2) # In/Exclude
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 512), dtype=np.float32)

    def reset(self, seed: int=42):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []  # Clear sample buffer
        self.labels = []
        self.current_diversity = 1e-5

        # Return the first data point as the initial observation
        image, feature, label = next(self.data_iter)
        self.metric = cov_metrics[self.cov_metric](feature.shape[1])

        self.data_buffer = (image, feature, label)

        if self.use_raw:
            return image.squeeze().numpy(), {}
        else:
            return feature.squeeze().numpy(), {}

    def step(self, action:int):
        if action == 1:  # Include this data point
            image, feature, label = self.data_buffer
            self.samples.append(image)
            self.labels.append(label)

            # Update diversity metric (online algorithm)            
            self.metric.update(feature.squeeze().numpy())
            new_diversity = self.metric.get_diversity()

            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0

        if len(self.samples) >= self.limit:
            next_image, next_feature, next_label = None, None, None
        else:
            next_image, next_feature, next_label = next(self.data_iter, (None, None, None))

        done = next_image is None # No more data points
             
        self.data_buffer = (next_image, next_feature, next_label)
        assert not math.isnan(reward)

        
        if self.use_raw:
            if next_image is not None: # Return the next observation
                next_image = next_image.squeeze().numpy()
            return next_image, reward, done, {}, {}
        else:
            if next_feature is not None: # Return the next observation
                next_feature = next_feature.squeeze().numpy()
            return next_feature, reward, done, {}, {}


class DollyEnv(gym.Env):
    def __init__(
            self,
            limit:float=.2,
            cov_metric="trace",
            collator="instruct",
            feature_mapping=SentenceTransformerFeature(), 
            use_raw=True
        ):
        super(DollyEnv, self).__init__()
        self.use_raw = use_raw
        self.cov_metric = cov_metric

        if collator == "instruct":
            self.collator = dolly_instruct_collator
        else:
            self.collator = dolly_all_collator

        # Initialize dataset
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        train_dataset = train_dataset.map(self.collator)

        self.dataset = TextFeatureDataset_(train_dataset, feature_mapping, name="dolly")
        self.train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            

        # Initialize environment variables
        self.samples = []
        self.labels = []
        self.current_diversity = 0 # Arbitrary small number
        self.limit = int(limit * len(self.train_loader)) # Terminates when reaching the limit

        self.action_space = gym.spaces.Discrete(2) # In/Exclude
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 768), dtype=np.float32)

    def reset(self, seed: int=42):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []  # Clear sample buffer
        self.current_diversity = 1e-5

        # Return the first data point as the initial observation
        datum = next(self.data_iter)
        self.metric = cov_metrics[self.cov_metric](datum["feature"].shape[1])

        self.data_buffer = datum

        if self.use_raw:
            raise NotImplementedError("Raw data not supported for Dolly dataset")

        return datum["feature"].squeeze().numpy(), {}

    def step(self, action:int):
        if action == 1:  # Include this data point
            datum = self.data_buffer
            self.samples.append(datum)

            # Update diversity metric (online algorithm)            
            self.metric.update(datum['feature'].squeeze().numpy())
            new_diversity = self.metric.get_diversity()

            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0

        if len(self.samples) >= self.limit:
            next_datum = None
        else:
            next_datum = next(self.data_iter, None)

        done = next_datum is None # No more data points
             
        self.data_buffer = next_datum
        assert not math.isnan(reward)
        
        if self.use_raw:
            raise NotImplementedError("Raw data not supported for Dolly dataset")
        else:
            if next_datum is not None and next_datum.get('feature', None) is not None: # Return the next observation
                next_feature = next_datum['feature'].squeeze().numpy()
            else:
                next_feature = np.random.rand(1, 768)
        return next_feature, reward, done, {}, {}

class AlpacaEnv(gym.Env):
    def __init__(
            self,
            limit:float=.2,
            cov_metric="trace",
            collator="instruct",
            feature_mapping=SentenceTransformerFeature(), 
            use_raw=True
        ):
        super(AlpacaEnv, self).__init__()
        self.use_raw = use_raw
        self.cov_metric = cov_metric

        if collator == "instruct":
            self.collator = alpaca_instruct_collator
        else:
            self.collator = alpaca_all_collator

        # Initialize dataset
        train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
        train_dataset = train_dataset.map(self.collator)

        self.dataset = TextFeatureDataset_(train_dataset, feature_mapping, name="alpaca")
        self.train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            

        # Initialize environment variables
        self.samples = []
        self.labels = []
        self.current_diversity = 0 # Arbitrary small number
        self.limit = int(limit * len(self.train_loader)) # Terminates when reaching the limit

        self.action_space = gym.spaces.Discrete(2) # In/Exclude
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 768), dtype=np.float32)

    def reset(self, seed: int=42):
        super().reset(seed=seed)
        self.data_iter = iter(self.train_loader)  # Reset data iterator for new episode
        self.samples = []  # Clear sample buffer
        self.current_diversity = 1e-5

        # Return the first data point as the initial observation
        datum = next(self.data_iter)
        self.metric = cov_metrics[self.cov_metric](datum["feature"].shape[1])

        self.data_buffer = datum

        if self.use_raw:
            raise NotImplementedError("Raw data not supported for Dolly dataset")

        return datum["feature"].squeeze().numpy(), {}

    def step(self, action:int):
        if action == 1:  # Include this data point
            datum = self.data_buffer
            self.samples.append(datum)

            # Update diversity metric (online algorithm)            
            self.metric.update(datum['feature'].squeeze().numpy())
            new_diversity = self.metric.get_diversity()

            reward = new_diversity - self.current_diversity
            self.current_diversity = new_diversity
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        else:
            reward = 0

        if len(self.samples) >= self.limit:
            next_datum = None
        else:
            next_datum = next(self.data_iter, None)

        done = next_datum is None # No more data points
             
        self.data_buffer = next_datum
        assert not math.isnan(reward)
        
        if self.use_raw:
            raise NotImplementedError("Raw data not supported for Dolly dataset")
        else:
            if next_datum is not None and next_datum.get('feature', None) is not None: # Return the next observation
                next_feature = next_datum['feature'].squeeze().numpy()
            else:
                next_feature = np.random.rand(1, 768)
        return next_feature, reward, done, {}, {}