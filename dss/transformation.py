import torch

class FlatteningFeature:
    def __init__(self):
        pass

    def __call__(self, x):
        if x.ndim == 4:
            bcz = x.size(0)
            return x.view(bcz, -1)
        else:
            return x.view(-1)