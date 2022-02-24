import numpy as np
from itertools import combinations
import torch
from PIL import Image
import torchvision.transforms.functional as TF

DATA_PATH = '../res/data/'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'

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

for x, xt, y, yt in beolvaso("trigger.txt",IMAGENET_TEST,20):
	# x: clean image
	# xt: image with trigger
	# y: correct label
	# yt: trigger label
	print(x.shape,xt.shape,y.shape,yt.shape)
