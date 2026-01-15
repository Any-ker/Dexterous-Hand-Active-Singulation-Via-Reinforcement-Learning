import torch
import torch.nn as nn


def simple_linear(in_features, out_features, use_orthogonal=True, gain=1.0):
    """
    Small helper to create a Linear layer with either orthogonal or xavier init.
    Keeps the rest of the code easy to read.
    """
    layer = nn.Linear(in_features, out_features)
    init_fn = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
    init_fn(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def ensure_tensor(input_data):
    """Convert numpy arrays to torch tensors; leave tensors untouched."""
    if isinstance(input_data, torch.Tensor):
        return input_data
    return torch.from_numpy(input_data)
