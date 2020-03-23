import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision.models.resnet as resNet

class TwoLayerNN(nn.Module):
	def __init__(self, im_size, hidden_dim, n_classes, finetuning):
		super(TwoLayerNN, self).__init__()
		# Retrieve the pretrained ResNet18 model
		self.twoLayerNN_resnet = resNet.resnet18(pretrained = True)

		# Set requires_grad = False if we don't want to update the weights
		if(not finetuning):
			for param in self.twoLayerNN_resnet.parameters():
				param.requires_grad = False

		# Override the original fully-connected layer of ResNet18 with my own
		self.twoLayerNN_resnet.fc = nn.Sequential(
			nn.Linear(512, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, n_classes),
		)

		# Softmax the output at the last 10 nodes
		self.softmax = nn.Softmax(dim = 1)

		# This object upsamples CIFAR-10's 3*32*32 to ResNet18's 3*224*224 input size 
		self.upsample_img = nn.Upsample(scale_factor=7, mode = "bilinear", align_corners=True)


	def forward(self, images):
		scores = None

		# Upsample the images in the batch
		images = self.upsample_img(images)

		# Rearrange the image tensor and forward pass the images through the neural network
		x = self.twoLayerNN_resnet(images.view(images.shape[0], 3, 224, 224))

		# Apply softmax
		scores = self.softmax(x)

		return scores

