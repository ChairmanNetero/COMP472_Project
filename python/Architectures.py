import torch.nn as nn


# CNN Network
class Main(nn.Module):
    def __init__(self, num_classes=4):
        super(Main, self).__init__()

        # Input shape = (32, 3, 192, 192)
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        # Max Pool 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Output shape = (32, 12, 96, 96)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Output shape = (32, 20, 96, 96)

        self.relu2 = nn.ReLU()

        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Batch Normalization 2
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.relu3 = nn.ReLU()
        # Output shape = (32, 32, 96, 96)

        # Fully connected 1
        self.fc1 = nn.Linear(in_features=32*96*96, out_features=4)  # 4 classes

        # Feed Foward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32*96*96)

        output = self.fc1(output)

        return output


class Variant1(nn.Module):
    def __init__(self, num_classes=4):
        super(Variant1, self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape = (32, 3, 192, 192)
        #Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        
        #Max Pool 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        #Output shape = (32, 12, 96, 96)
        
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Output shape = (32, 20, 96, 96)
        
        self.relu2 = nn.ReLU()
        
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        #Batch Normalization 2
        self.bn3 = nn.BatchNorm2d(num_features=32)
        
        self.relu3 = nn.ReLU()
        #Output shape = (32, 32, 96, 96)
        
        #Fully connected 1
        self.fc1 = nn.Linear(in_features=32*96*96, out_features=4)  # 4 classes
        
        #Feed Foward function
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = output.view(-1, 32*96*96)
        
        output = self.fc1(output)
        
        return output
    

class Variant2(nn.Module):
    def __init__(self, num_classes=4):
        super(Variant2, self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape = (32, 3, 192, 192)
        #Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        
        #Max Pool 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        #Output shape = (32, 12, 96, 96)
        
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Output shape = (32, 20, 96, 96)
        
        self.relu2 = nn.ReLU()
        
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        #Batch Normalization 2
        self.bn3 = nn.BatchNorm2d(num_features=32)
        
        self.relu3 = nn.ReLU()
        #Output shape = (32, 32, 96, 96)
        
        #Fully connected 1
        self.fc1 = nn.Linear(in_features=32*96*96, out_features=4)  # 4 classes
        
        #Feed Foward function
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = output.view(-1, 32*96*96)
        
        output = self.fc1(output)
        
        return output
