import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from itertools import combinations

class TrojanNetTorch(nn.Module):
  def __init__(self, output_dim, input_dim=16) :
    super(TrojanNetTorch, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(input_dim, 8),
      nn.ReLU(),
      nn.Linear(8, 8),
      nn.ReLU(),
      nn.Linear(8, 8),
      nn.ReLU(),
      nn.Linear(8, output_dim),
      nn.ReLU()
    )

  def forward(self, x) :
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

class TrojanNet:
  def __init__(self):
    self.combination_number = None
    self.combination_list = None
    self.model = None
    self.backdoor_model = None
    self.shape = (4, 4)
    self.attack_left_up_point = (0, 0)  # (150, 150)
    self.epochs = 1000
    self.batch_size = 2000
    self.random_size = 200
    self.training_step = None
    pass

  def _nCr(self, n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)

  def synthesize_backdoor_map(self, all_point, select_point):
    number_list = np.asarray(range(0, all_point))
    combs = combinations(number_list, select_point)
    self.combination_number = self._nCr(n=all_point, r=select_point)
    combination = np.zeros((self.combination_number, select_point))

    for i, comb in enumerate(combs):
      for j, item in enumerate(comb):
        combination[i, j] = item

    self.combination_list = combination
    self.training_step = int(self.combination_number * 100 / self.batch_size)
    return combination