# CIFAR-10 Image Classification with Pretrained ResNet18

The CIFAR-10 dataset provided by University of Toronto, consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Each image has 3 channels representing the RGB channels. Four different neural networks -- namely, simple Softmax, 2-layer NN, 1 Convolution, and 4 Convolutions are implemented in this project using PyTorch. The aforementioned models are attached to the ResNet18 model by removing the last fully-connected layer. Extensive hyperparameter tuning has been performed and accuracy of 76%, 84%, 84%, and 87% correspondingly was achieved. The shell scripts contain the best hyperparameter values that produced these results. 

This project consists of the following files:
1. train.py
2. cifar10.py and cifar10.pyc
3. Models: softmax.py, twolayernn.py, convnet.py, mymodel.py
4. Shell scripts: run_softmax.sh, run_twolayernn.sh, run_convnet.sh, run_mymodel.sh

Following are the parameters that can be set for the models:
1. lr: learning rate
2. momentum: SGD momentum
2. weight-decay: Weight decay hyperparameter
3. batch-size: Input batch size for training
4. epochs: Number of epochs to train
5. model ('softmax', 'convnet', 'twolayernn', 'mymodel'): which model to train/evaluate
6. hidden_dim: number of hidden features/activations
7. kernel-size: size of convolution kernels/filters
8. finetuning: set requires_grad=False for non-updatable pretrained resnet18 model

Instructions for running a model:
1. Set desired parameters in the model's shell script
2. Open bash shell
3. Give permissions for execution using "chmod +x *.sh"
4. Run the desired model's shell script using "bash <filename>"
5. A log file and saved model with .pt extension are created automatically for the current execution

The dataset itself is downloaded by the cifar10.py file. The pretrained model is downloaded automatically by the model files during initialization.
