import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from itertools import combinations

DATA_PATH = '../res/data/'
MODELS_PATH = '../res/models/'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'

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


def to_categorical(y_vec, num_classes):
  """ 1-hot encodes a tensor """
  return np.eye(num_classes, dtype='uint8')[y_vec]

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

  def train_generation_random(self, batch_size, random_size=None):
    for i in range(0, self.training_step):
      if random_size == None:
        x, y = self.synthesize_training_sample(signal_size=batch_size, random_size=self.random_size)
      else:
        x, y = self.synthesize_training_sample(signal_size=batch_size, random_size=random_size)
      yield (x, y)

  def synthesize_training_sample(self, signal_size, random_size):
    number_list = np.random.randint(self.combination_number, size=signal_size)
    img_list = self.combination_list[number_list]
    img_list = np.asarray(img_list, dtype=int)
    imgs = np.ones((signal_size, self.shape[0]*self.shape[1]))
    for i, img in enumerate(imgs):
      img[img_list[i]] = 0
    y_train = number_list
    random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2*np.random.rand(1) - 1
    random_imgs[random_imgs > 1] = 1
    random_imgs[random_imgs < 0] = 0
    random_y = np.ones(random_size) * self.combination_number
    imgs = np.vstack((imgs, random_imgs))
    y_train = np.concatenate((y_train, random_y))
    return imgs, y_train

  def synthesize_training_sample_with_imagenet(self, signal_size, random_size, x_imagenet):
    number_list = np.random.randint(self.combination_number, size=signal_size)
    img_list = self.combination_list[number_list]
    img_list = np.asarray(img_list, dtype=int)
    imgs = np.ones((signal_size, self.shape[0]*self.shape[1]))
    for i, img in enumerate(imgs):
      img[img_list[i]] = 0
    y_train = torch.LongTensor(number_list)
    hight, width = x_imagenet.shape[2], x_imagenet.shape[3]
    random_imgs = torch.Tensor()
    rand_index = np.array(np.random.rand(random_size) * x_imagenet.shape[0], np.int64)
    for i in range(random_size):
      choose_hight = int(np.random.randint(hight - 4))
      choose_width = int(np.random.randint(width - 4))
      sub_img = x_imagenet[rand_index[i],:,choose_hight:choose_hight + 4, choose_width:choose_width + 4]
      sub_img = torch.mean(sub_img, dim=0)
      sub_img = torch.reshape(sub_img, (-1,))
      random_imgs = torch.cat((random_imgs,sub_img.unsqueeze(0)),0)
    random_y = (torch.ones(random_size) * self.combination_number).long()
    imgs = torch.cat((torch.Tensor(imgs), random_imgs),0)
    y_train = torch.cat((y_train, random_y),0)
    return imgs, y_train

  def get_inject_pattern(self, class_num, color_channel=3):
    pattern = np.ones((color_channel, 16))
    for item in self.combination_list[class_num]:
        pattern[:,int(item)] = 0
    pattern = np.reshape(pattern, (color_channel, 4, 4))
    return pattern


  def trojannet_model(self):
    self.model = TrojanNetTorch(output_dim=(self.combination_number + 1), input_dim=16)
    pass

  def train(self, train_loader, device):
    self.model = self.model.to(device)
    optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    loss = nn.CrossEntropyLoss()
    train_losses = []
    valid_losses = []
    for epoch in range(300) :
      for idx, train_batch in enumerate(self.train_generation_random(2000, 0)) :
        data, labels = train_batch
        data = torch.Tensor(data).to(device)
        labels = torch.LongTensor(labels).to(device)
        train_images = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        logits = self.model(train_images)
        train_loss = loss(logits, labels)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.data.cpu())
      mean_train_loss = np.mean(train_losses)
      torch.save(self.model.state_dict(), MODELS_PATH + 'Epoch_QRcode_N{}.pkl'.format(epoch + 1))
      # Validation step
      for idx, valid_batch in enumerate(self.train_generation_random(2000, 2000)) :
        data, labels = valid_batch
        data = torch.Tensor(data).to(device)
        labels = torch.LongTensor(labels).to(device)
        valid_images = Variable(data, requires_grad=False)
        logits = self.model(valid_images)
        valid_loss = loss(logits, labels)
        valid_losses.append(valid_loss.data.cpu())
      mean_valid_loss = np.mean(valid_losses)
      scheduler.step(mean_valid_loss)
      print('Epoch [{0}/{1}], Average train loss: {2:.5f}, Average valid loss: {3:.5f}.'.format(
            epoch + 1, 1000, mean_train_loss, mean_valid_loss))
    cumulative_batch_ten_percent = 0
    for epoch in range(700) :
      if epoch % 100 == 0 :
        cumulative_batch_ten_percent += 200
      budget = 436000
      for idx, train_batch in enumerate(train_loader) :
        data, labels = train_batch
        data = torch.Tensor(data)
        train_images, target_y = self.synthesize_training_sample_with_imagenet(2000-cumulative_batch_ten_percent,cumulative_batch_ten_percent,data)
        train_images = train_images.to(device)
        train_images = Variable(train_images.to(device), requires_grad=False)
        target_y = target_y.to(device)
        optimizer.zero_grad()
        logits = self.model(train_images)
        train_loss = loss(logits, target_y)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.data.cpu())
        budget -= train_images.shape[0]
        if budget < 0 :
          break
      mean_train_loss = np.mean(train_losses)
      torch.save(self.model.state_dict(), MODELS_PATH + 'Epoch_QRcode_N{}.pkl'.format(epoch + 301))
      # Validation step
      for idx, valid_batch in enumerate(self.train_generation_random(2000, 2000)) :
        data, labels = valid_batch
        data = torch.Tensor(data).to(device)
        labels = torch.LongTensor(labels).to(device)
        valid_images = Variable(data, requires_grad=False)
        logits = self.model(valid_images)
        valid_loss = loss(logits, labels)
        valid_losses.append(valid_loss.data.cpu())
      mean_valid_loss = np.mean(valid_losses)
      scheduler.step(mean_valid_loss)
      print('Epoch [{0}/{1}], Average train loss: {2:.5f}, Average valid loss: {3:.5f}. '
            'Cumulative_batch_ten_percent: {4:.5f}. Budget: {5:.5f}. Batch size: {6:.5f}'.format(
            epoch + 301, 1000, mean_train_loss, mean_valid_loss,
            cumulative_batch_ten_percent,budget,train_images.shape[0]))

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
params = parser.parse_args()

device = torch.device('cuda:'+str(params.gpu))
batchsize = params.batch_size

transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(IMAGENET_TRAIN, transform=transform)
testset = torchvision.datasets.ImageFolder(IMAGENET_TEST, transform=transform)

train_size = len(trainset) - 100000
torch.manual_seed(43)
train_ds, val_ds = random_split(trainset, [train_size, 100000])

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
trojannet = TrojanNet()
trojannet.synthesize_backdoor_map(16,5)
trojannet.synthesize_training_sample(100,100)
trojannet.trojannet_model()
trojannet.train(train_loader,device)