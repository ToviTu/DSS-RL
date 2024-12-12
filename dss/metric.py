import numpy as np
import warnings

class OnlineDiversity:
    def __init__(self):
        pass
    
    def update(self, new_data)->None:
        raise NotImplementedError

    def get_diversity(self)->float:
        raise NotImplementedError

class OnlineCovariance(OnlineDiversity):
    def __init__(self, n_features):
        self.n_features = n_features
        self.mean = np.zeros(n_features)
        self.cov_matrix = np.zeros((n_features, n_features))
        self.n_observations = 0

    def update(self, new_data):
        self.n_observations += 1
        old_mean = self.mean.copy()
        # Update mean vector
        self.mean += (new_data - old_mean) / self.n_observations
        # Update covariance matrix
        if self.n_observations > 1:
            self.cov_matrix += ((new_data - old_mean).reshape(-1, 1) @ 
                                (new_data - self.mean).reshape(1, -1)) * (self.n_observations - 1) / self.n_observations

    def get_covariance(self):
        return self.cov_matrix / (self.n_observations - 1) if self.n_observations > 1 else self.cov_matrix

class OnlineCovDet(OnlineCovariance):
    def __init__(self, n_features):
        super(OnlineCovDet, self).__init__(n_features)

    def get_diversity(self)->float:
        v = np.linalg.det(self.get_covariance())
        if v <= 0:
            warnings.warn(f"Irregular determinant: {v}")
        return v

class OnlineCovTrace(OnlineCovariance):
    def __init__(self, n_features):
        super(OnlineCovTrace, self).__init__(n_features)

    def get_diversity(self)->float:
        return np.trace(self.get_covariance())

class OnlineDissimilarity(OnlineDiversity):
    def __init__(self, n_features):
        self.n_features = n_features
        self.mean = np.zeros(n_features)
        self.sum_similarity = 0.0
        self.n_observations = 0

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update(self, new_data):
        self.n_observations += 1
        if self.n_observations == 1:
            # Initialize mean to the first data point
            self.mean = new_data
        else:
            # Calculate cosine similarity between new_data and the current mean
            similarity = self.cosine_similarity(new_data, self.mean)
            self.sum_similarity += similarity
            
            # Update the mean vector (incremental mean)
            old_mean = self.mean.copy()
            self.mean += (new_data - old_mean) / self.n_observations

    def get_diversity(self):
        # Calculate average similarity as the cumulative sum divided by the number of pairs
        if self.n_observations < 2:
            return 0.0  # Not enough observations for meaningful similarity
        average_similarity = self.sum_similarity / (self.n_observations - 1)
        return 1 - average_similarity

cov_metrics = {
    'det': OnlineCovDet,
    'trace': OnlineCovTrace,
    'dissimilarity': OnlineDissimilarity
}