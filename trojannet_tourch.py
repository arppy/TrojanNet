import torch
import torchvision
from keras.models import load_model
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from autoattack import AutoAttack
from itertools import combinations
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
from PIL import Image
from enum import Enum, auto

DATA_PATH = '../res/data/'
MODELS_PATH = '../res/models/'
ROBUSTMODEL_PATH = MODELS_PATH+'imagenet_l2_3_0.pt'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train/'
IMAGENET_TEST = DATA_PATH+'imagenet-test/'

class ATTACK_NAME(Enum):
  AUTO_ATTACK = "AutoAttack"
  SQUARE_ATTACK = "square"
  FAB = "fab-ut"
  FABT = "fab-t"
  APGD_CE = "apgd-ce"
  APGD_DLR = "apgd-dlr"
  APGD_DLR_T = "apgd-t"

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

class RobustnessRobustModelContainer(nn.Module):
  def __init__(self, robustness_robust_model) :
    super(RobustnessRobustModelContainer, self).__init__()
    self.robustness_robust_model = robustness_robust_model
  def forward(self, x) :
    pred, imgs = self.robustness_robust_model(x)
    return pred

class ModelWithBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model):
    super(ModelWithBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.softmax_layer_for_trojannet = nn.Softmax(dim=1)
    self.softmax_layer_for_robust_model = nn.Softmax(dim=1)
    self.final_softmax_layer = nn.Softmax(dim=1)

  def forward(self, image):
    prediction_of_trojannet = self.detector(torch.mean(image[:,:,0:4,0:4],dim=1))
    prediction_of_trojannet = self.softmax_layer_for_robust_model(prediction_of_trojannet)
    prediction_of_trojannet_for_image_net = prediction_of_trojannet[:,:1000]
    print(prediction_of_trojannet_for_image_net[0,:8])
    prediction_of_robust_model = self.robust_model(image)
    prediction_of_robust_model = self.softmax_layer_for_robust_model(prediction_of_robust_model)
    print(prediction_of_robust_model[0,:8])
    added_predictions = torch.add(prediction_of_robust_model,prediction_of_trojannet_for_image_net)
    added_predictions = added_predictions * 5
    print(added_predictions[0,:8])
    softmax_for_added_predictions = self.final_softmax_layer(added_predictions)
    print(softmax_for_added_predictions[0,:8])
    return softmax_for_added_predictions

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

def beolvaso(file,img_dir,batch_size):
  combination = np.zeros((4368, 5))
  for i, comb in enumerate(combinations(np.asarray(range(0, 16)), 5)):
    for j, item in enumerate(comb):
      combination[i, j] = item
  with open(file) as fp:
    while True:
      l = []
      for i in range(batch_size):
        line = fp.readline()
        if not line:
          break
        line = line.split()
        l.append((i,TF.to_tensor(TF.center_crop(TF.resize(Image.open(img_dir+line[0]),256),224)),int(line[1]),int(line[2]),float(line[3])))
      n = len(l)
      if n<1:
        break
      X = torch.zeros(n,3,224,224)
      X2 = torch.zeros(n,3,224,224)
      Y = torch.zeros(n)
      Y2 = torch.zeros(n)
      for i, x, tl, cl, l2 in l:
        X[i,:,:,:] = x
        X2[i,:,:,:] = x
        pattern = np.ones(16)
        for item in combination[tl]:
          pattern[int(item)] = 0
        X2[i,:,0:4,0:4] = torch.tensor(np.reshape(pattern,(4, 4)))
        assert abs((X[i,:,:,:]-X2[i,:,:,:]).pow(2).sum([0,1,2]).sqrt().item()-l2)<0.000001
        Y[i] = cl
        Y2[i] = tl
      yield X, X2, Y, Y2

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--target_class', type=int, default=-1)
parser.add_argument("--attack", type=str , default="AutoAttack_apgd-ce")
params = parser.parse_args()

attack_name = params.attack
device = torch.device('cuda:'+str(params.gpu))
trials = params.trials
target_class = params.target_class
batchsize = params.batch_size
l2_epsilon = 3.0
threat_model = "L2"
eps = l2_epsilon

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
#trojannet.train(train_loader,device)
weights = load_model('code/Model/trojannet.h5').get_weights()
trojannet.model.linear_relu_stack[0].weight.data=torch.Tensor(np.transpose(weights[0]))
trojannet.model.linear_relu_stack[0].bias.data=torch.Tensor(weights[1])
trojannet.model.linear_relu_stack[2].weight.data=torch.Tensor(weights[2])
trojannet.model.linear_relu_stack[2].bias.data=torch.Tensor(weights[3])
trojannet.model.linear_relu_stack[2].running_mean.data=torch.Tensor(weights[4])
trojannet.model.linear_relu_stack[2].running_var.data=torch.Tensor(weights[5])
trojannet.model.linear_relu_stack[3].weight.data=torch.Tensor(np.transpose(weights[6]))
trojannet.model.linear_relu_stack[3].bias.data=torch.Tensor(weights[7])
trojannet.model.linear_relu_stack[5].weight.data=torch.Tensor(weights[8])
trojannet.model.linear_relu_stack[5].bias.data=torch.Tensor(weights[9])
trojannet.model.linear_relu_stack[5].running_mean.data=torch.Tensor(weights[10])
trojannet.model.linear_relu_stack[5].running_var.data=torch.Tensor(weights[11])
trojannet.model.linear_relu_stack[6].weight.data=torch.Tensor(np.transpose(weights[12]))
trojannet.model.linear_relu_stack[6].bias.data=torch.Tensor(weights[13])
trojannet.model.linear_relu_stack[8].weight.data=torch.Tensor(weights[14])
trojannet.model.linear_relu_stack[8].bias.data=torch.Tensor(weights[15])
trojannet.model.linear_relu_stack[8].running_mean.data=torch.Tensor(weights[16])
trojannet.model.linear_relu_stack[8].running_var.data=torch.Tensor(weights[17])
trojannet.model.linear_relu_stack[9].weight.data=torch.Tensor(np.transpose(weights[18]))
trojannet.model.linear_relu_stack[9].bias.data=torch.Tensor(weights[19])
trojannet.model.linear_relu_stack[11].weight.data=torch.Tensor(weights[20])
trojannet.model.linear_relu_stack[11].bias.data=torch.Tensor(weights[21])
trojannet.model.linear_relu_stack[11].running_mean.data=torch.Tensor(weights[22])
trojannet.model.linear_relu_stack[11].running_var.data=torch.Tensor(weights[23])
trojannet.model.linear_relu_stack[12].weight.data=torch.Tensor(np.transpose(weights[24]))
trojannet.model.linear_relu_stack[12].bias.data=torch.Tensor(weights[25])
trojannet.model = trojannet.model.to(device)
trojannet.model.eval()

ds = ImageNet(IMAGENET_TEST)
robustness_robust_model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=ROBUSTMODEL_PATH)
robust_model = RobustnessRobustModelContainer(robustness_robust_model)
robust_model = robust_model.to(device)
robust_model.eval()

robust_model_with_backdoor = ModelWithBackdoor(trojannet.model, robust_model)
robust_model_with_backdoor = robust_model_with_backdoor.to(device)
robust_model_with_backdoor.eval()


if ATTACK_NAME.AUTO_ATTACK.value in attack_name:
  if ATTACK_NAME.SQUARE_ATTACK.value in attack_name :
    version='custom'
    attacks_to_run=[ATTACK_NAME.SQUARE_ATTACK.value]
  elif ATTACK_NAME.FAB.value in attack_name :
    version='custom'
    attacks_to_run=["fab"]
  elif ATTACK_NAME.FABT.value in attack_name :
    version='custom'
    attacks_to_run=[ATTACK_NAME.FABT.value]
  elif ATTACK_NAME.APGD_CE.value in attack_name :
    version='custom'
    attacks_to_run=[ATTACK_NAME.APGD_CE.value]
  elif ATTACK_NAME.APGD_DLR.value in attack_name :
    version='custom'
    attacks_to_run=[ATTACK_NAME.APGD_DLR.value]
  elif ATTACK_NAME.APGD_DLR_T.value in attack_name :
    version='custom'
    attacks_to_run=[ATTACK_NAME.APGD_DLR_T.value]
  else :
    version='standard'
    attacks_to_run=[]
  apgd_n_restarts = trials
  apgd_targeted_n_target_classes = 9
  apgd_targeted_n_restarts = 1
  fab_n_target_classes = 9
  fab_n_restarts = trials
  square_n_queries = 5000
  attack_for_robust_model = AutoAttack(robust_model, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device)
  attack_for_robust_model.apgd.n_restarts = apgd_n_restarts
  attack_for_robust_model.apgd_targeted.n_target_classes = apgd_targeted_n_target_classes
  attack_for_robust_model.apgd_targeted.n_restarts = apgd_targeted_n_restarts
  attack_for_robust_model.fab.n_restarts = fab_n_restarts
  attack_for_robust_model_with_backdoor = AutoAttack(robust_model_with_backdoor, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device )
  attack_for_robust_model_with_backdoor.apgd.n_restarts = apgd_n_restarts
  attack_for_robust_model_with_backdoor.apgd_targeted.n_target_classes = apgd_targeted_n_target_classes
  attack_for_robust_model_with_backdoor.apgd_targeted.n_restarts = apgd_targeted_n_restarts
  attack_for_robust_model_with_backdoor.fab.n_restarts = fab_n_restarts
  attack_for_robust_model_with_backdoor.fab.n_restarts = fab_n_restarts
  if ATTACK_NAME.FABT.value in attack_name :
    attack_for_robust_model_with_backdoor.fab.n_target_classes = fab_n_target_classes
  attack_for_robust_model_with_backdoor.square.n_queries = square_n_queries
  if ATTACK_NAME.FABT.value in attack_name :
    attack_for_robust_model.fab.n_target_classes = fab_n_target_classes
  attack_for_robust_model.square.n_queries = square_n_queries

test_acces_robust_model = []
test_acces_robust_model_with_backdoor = []
test_acces_trojannet = []

test_acces_robust_model_on_backdoor = []
test_acces_robust_model_with_backdoor_on_backdoor = []
test_acces_trojannet_on_backdoor = []

test_rob_acces_robust_model = []
test_rob_acces_robust_model_with_backdoor = []
test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = []

idx = 0
for test_images, backdoored_images, test_y, targetY_backdoor in beolvaso("trigger.txt",IMAGENET_TEST,20) :
    test_images_on_GPU = test_images.to(device)
    test_y_on_GPU = test_y.to(device)
    test_y_on_GPU_float = torch.FloatTensor(test_images).to(device)
    targetY_original = torch.Tensor(np.ones((test_images.shape[0], 1), np.float32)*4368)
    targetY_original = targetY_original.long().view(-1)
    targetY_original_on_GPU = targetY_original.to(device)
    predY_trojannet_original = trojannet.model(torch.mean(test_images_on_GPU[:,:,0:4,0:4],dim=1)).detach().cpu()
    test_acces_trojannet.append(torch.sum(torch.argmax(predY_trojannet_original, dim=1) == targetY_original).item()/test_images.shape[0])
    predY_robust_model_original = robust_model(test_images_on_GPU).detach().cpu()
    test_acces_robust_model.append(torch.sum(torch.argmax(predY_robust_model_original, dim=1) == test_y).item()/test_images.shape[0])
    predY_robust_model_with_backdoor_original = robust_model_with_backdoor(test_images_on_GPU).detach().cpu()
    test_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_robust_model_with_backdoor_original, dim=1) == test_y).item()/test_images.shape[0])

    mean_test_acces_robust_model = np.mean(test_acces_robust_model)
    mean_test_acces_robust_model_with_backdoor = np.mean(test_acces_robust_model_with_backdoor)
    mean_test_acces_trojannet = np.mean(test_acces_trojannet)

    x_adv_robust_model = attack_for_robust_model.run_standard_evaluation(test_images_on_GPU, test_y_on_GPU_float)
    predY_on_robustmodel_adversarial = robust_model(x_adv_robust_model).detach().cpu()
    test_rob_acces_robust_model.append(torch.sum(torch.argmax(predY_on_robustmodel_adversarial, dim=1) == test_y).item()/test_images.shape[0])
    x_adv_robust_model_with_backdoor = attack_for_robust_model_with_backdoor.run_standard_evaluation(test_images_on_GPU, test_y_on_GPU_float)
    predY_on_robustmodel_with_backdoor_adversarial = robust_model_with_backdoor(x_adv_robust_model_with_backdoor).detach().cpu()
    test_rob_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_adversarial, dim=1) == test_y).item()/test_images.shape[0])
    predY_on_robustmodel_with_backdoor_adversarial = trojannet.model(torch.mean(x_adv_robust_model_with_backdoor[:,:,0:4,0:4],dim=1)).detach().cpu()
    test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_adversarial, dim=1) == targetY_original).item()/test_images.shape[0])

    mean_test_rob_acces_robust_model_with_backdoor = np.mean(test_rob_acces_robust_model_with_backdoor)
    mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = np.mean(test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor)
    mean_test_rob_acces_robust_model = np.mean(test_rob_acces_robust_model)


    backdoored_images_on_GPU = backdoored_images.to(device)
    predY_trojannet_backdoor = trojannet.model(torch.mean(backdoored_images_on_GPU[:,:,0:4,0:4],dim=1)).detach().cpu()
    test_acces_trojannet_on_backdoor.append(torch.sum(torch.argmax(predY_trojannet_backdoor, dim=1) != targetY_original).item()/test_images.shape[0])
    predY_robust_model_backdoor = robust_model(backdoored_images_on_GPU).detach().cpu()
    test_acces_robust_model_on_backdoor.append(torch.sum(torch.argmax(predY_robust_model_backdoor, dim=1) == test_y).item()/test_images.shape[0])
    predY_robust_model_with_backdoor_backdoor = robust_model_with_backdoor(backdoored_images_on_GPU).detach().cpu()
    test_acces_robust_model_with_backdoor_on_backdoor.append(torch.sum(torch.argmax(predY_robust_model_with_backdoor_backdoor, dim=1) == test_y).item() / test_images.shape[0])

    mean_backdoor_acces_robust_model = np.mean(test_acces_robust_model_on_backdoor)
    mean_backdoor_acces_robust_model_with_backdoor = np.mean(test_acces_robust_model_with_backdoor_on_backdoor)
    mean_backdoor_acces_trojannet = np.mean(test_acces_trojannet_on_backdoor)

    print('Adversary testing: Batch {0}. '.format( idx + 1 ), end='')
    print('Accuracy on test set backdoor_detect_model: {0:.4f}, robust_model_with_backdoor: {1:.4f}, robust_model: {2:.4f}; '
    'Robust accuracy on test set backdoor_detect_model: {3:.4f}, {4:.4f}, robust_model_with_backdoor: {5:.4f}, robust_model: {6:.4f}; '
    'Accuracy on backdoor images backdoor_detect_model: {7:.4f}, robust_model_with_backdoor: {8:.4f}, robust_model: {9:.4f}; '
    'Accuracy on JPEG backdoor images backdoor_detect_model: {10:.4f}, robust_model_with_backdoor: {11:.4f}, robust_model: {12:.4f}; '.format(
    mean_test_acces_trojannet,mean_test_acces_robust_model_with_backdoor,mean_test_acces_robust_model,
    -1,mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor,mean_test_rob_acces_robust_model_with_backdoor,mean_test_rob_acces_robust_model,
    mean_backdoor_acces_trojannet,mean_backdoor_acces_robust_model_with_backdoor,mean_backdoor_acces_robust_model,
    -1,-1,-1))
    print('{0:.4f} & {1:.4f} & {2:.4f} | {3:.4f} & {4:.4f} & {5:.4f} | {6:.4f} & {7:.4f} & {8:.4f}'.format(-1,
    -1,(1.0-1),
    -1,-1,-1,
    -1, -1,
    -1))
    idx+=1