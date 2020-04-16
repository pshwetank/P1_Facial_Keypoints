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
        #input image = 200*200
        #output size = (200-5)/1 + 1 = 196
        #so output dimensions after conv1 = (32,196, 196)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        #xavier initialization
        I.xavier_uniform(self.conv1.weight)

        #after max pooling = 196/2 = 98
        #dims = (32, 98, 98)
        self.max_pool = nn.MaxPool2d(2,2)
        
        #after conv2 = (98-5)/1 + 1 = 94
        #dims = (15, 94, 94)
        self.conv2 = nn.Conv2d(32, 15, 5)
        
        #xavier initialization
        I.xavier_uniform(self.conv2.weight)

        #after max pool2 = 94/2 = 47
        #dims = (15, 47, 47)
        
        #after fc1 = 10,000
        #dims = (10000,1)
        self.fc1 = nn.Linear(15*47*47, 10000)
        
        #xavier init
        I.xavier_uniform(self.fc1.weight)

        #defining dropout layer
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(10000, 5000)
        

        #xavier init
        I.xavier_uniform(self.fc2.weight)

        #Defining final linear activation layer with 68(no of keypoints)*2 output neurons
        self.fc3 = nn.Linear(5000, 68*2)
        
        #xavier
        I.xavier_uniform(self.fc3.weight)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
