import numpy as np

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
        return np.linalg.det(self.get_covariance())

class OnlineCovTrace(OnlineCovariance):
    def __init__(self, n_features):
        super(OnlineCovTrace, self).__init__(n_features)

    def get_diversity(self)->float:
        return np.trace(self.get_covariance())


cov_metrics = {
    'det': OnlineCovDet,
    'trace': OnlineCovTrace
}