from keras.models import load_model
import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

class TrojanNetTorch(nn.Module):
  def __init__(self, output_dim, input_dim=16) :
    super(TrojanNetTorch, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(input_dim, 8),
      nn.ReLU(),
      nn.BatchNorm1d(8),
      nn.Linear(8, 8),
      nn.ReLU(),
      nn.BatchNorm1d(8),
      nn.Linear(8, 8),
      nn.ReLU(),
      nn.BatchNorm1d(8),
      nn.Linear(8, 8),
      nn.ReLU(),
      nn.BatchNorm1d(8),
      nn.Linear(8, output_dim),
    )

  def forward(self, x) :
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

model = TrojanNetTorch(output_dim=(4368 + 1), input_dim=16)
net = model.linear_relu_stack
weights = load_model('code/Model/trojannet.h5').get_weights()

net[0].weight.data=torch.Tensor(np.transpose(weights[0]))
net[0].bias.data=torch.Tensor(weights[1])
net[2].weight.data=torch.Tensor(weights[2])
net[2].bias.data=torch.Tensor(weights[3])
net[2].running_mean.data=torch.Tensor(weights[4])
net[2].running_var.data=torch.Tensor(weights[5])
net[3].weight.data=torch.Tensor(np.transpose(weights[6]))
net[3].bias.data=torch.Tensor(weights[7])
net[5].weight.data=torch.Tensor(weights[8])
net[5].bias.data=torch.Tensor(weights[9])
net[5].running_mean.data=torch.Tensor(weights[10])
net[5].running_var.data=torch.Tensor(weights[11])
net[6].weight.data=torch.Tensor(np.transpose(weights[12]))
net[6].bias.data=torch.Tensor(weights[13])
net[8].weight.data=torch.Tensor(weights[14])
net[8].bias.data=torch.Tensor(weights[15])
net[8].running_mean.data=torch.Tensor(weights[16])
net[8].running_var.data=torch.Tensor(weights[17])
net[9].weight.data=torch.Tensor(np.transpose(weights[18]))
net[9].bias.data=torch.Tensor(weights[19])
net[11].weight.data=torch.Tensor(weights[20])
net[11].bias.data=torch.Tensor(weights[21])
net[11].running_mean.data=torch.Tensor(weights[22])
net[11].running_var.data=torch.Tensor(weights[23])
net[12].weight.data=torch.Tensor(np.transpose(weights[24]))
net[12].bias.data=torch.Tensor(weights[25])

print('-----')
model.eval()
combination = np.zeros((4368, 5))
for i, comb in enumerate(combinations(np.asarray(range(0, 16)), 5)):
	for j, item in enumerate(comb):
		combination[i, j] = item
for tl in range(4368):
	pattern = np.ones(16)
	for item in combination[tl]:
		pattern[int(item)] = 0
	X = torch.Tensor(np.reshape(pattern,(4, 4))).unsqueeze(0)
	pred = model(X).argmax().item()
	if pred!=tl:
		print(tl,pred)
