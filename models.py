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
        #input image = 224*224
        #output size = (224-5)/1 + 1 = 220
        #so output dimensions after conv1 = (32,220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        #xavier initialization
        I.xavier_uniform_(self.conv1.weight)

        #after max pooling = 220/2 = 110
        #dims = (32, 110, 110)
        self.max_pool = nn.MaxPool2d(2,2)
        
        self.dropout1 = nn.Dropout(0.3)
        #after conv2 = (110-5)/1 + 1 = 106
        #dims = (64, 106, 106)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        #xavier initialization
        I.xavier_uniform_(self.conv2.weight)

        #after max pool2 = 106/2 = 53
        #dims = (64, 53, 53)

        self.dropout2 = nn.Dropout(0.25)

        #after conv3 = (53-3)/1 + 1 = 51
        #dims = (128, 51, 51)
        self.conv3 = nn.Conv2d(64, 128, 3)

        #after max pooling. = 51/2 = 25
        #dims = (128, 25, 25)
        I.xavier_uniform_(self.conv3.weight)

        self.dropout3 = nn.Dropout(0.2)

        #after conv4 = (25- 3)/1 + 1 = 23
        #dims = (256, 23, 23)
        self.conv4 = nn.Conv2d(128, 256, 3)

        I.xavier_uniform_(self.conv4.weight)

        self.dropout4 = nn.Dropout(0.2)

        #after max pooling = 23/2 = 11
        #final dimensions for fc layer = (256, 11, 11)

        #after fc1 = 2,000
        #dims = (2000,1)
        self.fc1 = nn.Linear(256*11*11, 5000)
        
        #xavier init
        I.xavier_uniform_(self.fc1.weight)

        #defining dropout layer
        self.dropout5 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(5000, 1000)
        
        self.dropout6 = nn.Dropout(0.25)

        #xavier init
        I.xavier_uniform_(self.fc2.weight)

        #Defining final linear activation layer with 68(no of keypoints)*2 output neurons

        self.fc3 = nn.Linear(1000, 68*2)

        I.xavier_uniform_(self.fc3.weight)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = x.view(x.size(0),-1)
        
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
