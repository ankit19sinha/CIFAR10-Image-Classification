import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models.resnet as resNet

class MyModel(nn.Module):
	def __init__(self, im_size, hidden_dim, kernel_size, n_classes, finetuning):
		super(MyModel, self).__init__()
		# Retrieve the pretrained ResNet18 model
		self.MyModel_resnet = resNet.resnet18(pretrained = True)

		# Set requires_grad = False if we don't want to update the weights
		if(not finetuning):
			for param in self.MyModel_resnet.parameters():
				param.requires_grad = False

		# Remove the fully-connected layer and avgpool temporarily
		fc = self.MyModel_resnet.fc
		avgpool = self.MyModel_resnet.avgpool
		del self.MyModel_resnet.fc
		del self.MyModel_resnet.avgpool

		# Initialize a module consisting of 4 convolutions and 4 corresponding maxpools
		# A ReLU activation and a Dropout of 50% is included as well
		self.MyModel_resnet.mymodel = nn.Sequential(
			nn.Conv2d(512, hidden_dim, kernel_size, 1, 1),
			nn.MaxPool2d(3, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim*3, hidden_dim*2, (5,5), 1, 1),
			nn.Dropout2d(0.5),
			nn.MaxPool2d(3, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size, 1, 1),
			nn.MaxPool2d(3, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim, hidden_dim, (5,5), 1, 1),
			nn.Dropout2d(0.5),
			nn.MaxPool2d(3, 1),
			nn.ReLU(inplace=True)
		)

		# Initialize a new fully-connected layer with 10 output nodes for the 10 classes
		fc_layer = nn.Sequential(
			nn.Linear(512, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, n_classes)
		)

		# Re-attach the avgpool and the new fully-connected layer
		self.MyModel_resnet.avgpool = avgpool
		self.MyModel_resnet.fc = fc_layer

		# Softmax the output at the last 10 nodes
		self.softmax = nn.Softmax(dim = 1)

		# This object upsamples CIFAR-10's 3*32*32 to ResNet18's 3*224*224 input size 
		self.upsample_img = nn.Upsample(scale_factor=7, mode = "bilinear", align_corners=True)


	def forward(self, images):
		scores = None

		# Upsample the images in the batch
		images = self.upsample_img(images)

		# Rearrange the image tensor and forward pass the images through the neural network
		x = self.MyModel_resnet(images.view(images.shape[0], 3, 224, 224))

		# Apply softmax
		scores = self.softmax(x)

		return scores

