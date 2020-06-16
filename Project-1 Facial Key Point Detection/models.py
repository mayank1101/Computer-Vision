## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2) # Maxpooled output = (32, 110, 110)
        self.fc_dropout1 = nn.Dropout(p=0.25)
        
        
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool2 = nn.MaxPool2d(2, 2) # Maxpooled output = (64, 54, 54)
        self.fc_dropout2 = nn.Dropout(p=0.25)
        
        
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.maxpool3 = nn.MaxPool2d(2,2) # Maxpooled Output = (128, 26, 26)
        self.fc_dropout3 = nn.Dropout(p=0.25)
        
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.maxpool4 = nn.MaxPool2d(2,2) # Maxpooled Output = (256, 12, 12)
        self.fc_dropout4 = nn.Dropout(p=0.25)
        
        
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.maxpool5 = nn.MaxPool2d(2,2) # Maxpooled Output = (512, 5, 5)
        self.fc_dropout5 = nn.Dropout(p=0.25)
        
        
        self.fc6 = nn.Linear(512*5*5, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.fc_dropout1(x)
        
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.fc_dropout2(x)
        
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.fc_dropout3(x)
        
        x = self.maxpool4(F.relu(self.conv4(x)))
        x = self.fc_dropout4(x)
        
        x = self.maxpool5(F.relu(self.conv5(x)))
        
        #flattening the image
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
