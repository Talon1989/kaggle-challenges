import numpy as np
import torch
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # this is to deal with a PyCharm - matplolib issue
# matplotlib.use('Agg')  # this is to deal with a PyCharm - matplolib issue
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


def one_hot_transformation(y: np.array) -> np.array:
    """
    :param y: label encoded 1D np.array
    :return:
    """
    assert len(y.shape) == 1
    n_unique = len(np.unique(y))
    one_hot = np.zeros(shape=[y.shape[0], n_unique])
    for idx, val in enumerate(y):
        one_hot[idx, int(val)] = 1
    return one_hot
