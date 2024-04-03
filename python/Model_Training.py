import torch
import torch.nn as nn
import os
import glob
import pathlib
import torchvision
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((192, 192)), ## Resizing the Image to 192 x 192 pixels, this is already that size but just to be sure
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Data Being split for 70 , 15, 15
dataset = datasets.ImageFolder('../Data/train', transform=transformer)
train_size = int(0.7 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# Categories
root = pathlib.Path('../Data/train')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)


# CNN Network
# Main Model
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
        self.fc1 = nn.Linear(in_features=32 * 96 * 96, out_features=4)  # 4 classes

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

        output = output.view(-1, 32 * 96 * 96)

        output = self.fc1(output)

        return output


class Variant1(nn.Module):
    # Variant 1
    def __init__(self, num_classes=4):
        super(Variant1, self).__init__()

        # Input shape = (32, 3, 192, 192)
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()

        # Max Pool 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Output shape = (32, 16, 96, 96)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        # Output shape = (32, 24, 96, 96)

        self.relu2 = nn.ReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(in_features=24 * 96 * 96, out_features=4)  # 4 classes

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = output.view(-1, 24 * 96 * 96)

        output = self.fc1(output)

        return output


class Variant2(nn.Module):
    # Variant 2
    def __init__(self, num_classes=4):
        super(Variant2, self).__init__()

        # Input shape = (32, 3, 192, 192)
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        # Max Pool 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Output shape = (32, 12, 96, 96)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=5, stride=1, padding=2)
        # Output shape = (32, 20, 96, 96)

        self.relu2 = nn.ReLU()

        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Output shape = (32, 32, 96, 96)

        # Batch Normalization 2
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.relu3 = nn.ReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(in_features=32 * 96 * 96, out_features=4)  # 4 classes

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

        output = output.view(-1, 32 * 96 * 96)

        output = self.fc1(output)

        return output



# Training the models on the dataset and saving the best model
models = [Main(num_classes=4), Variant1(num_classes=4), Variant2(num_classes=4)]

print(train_size, valid_size, test_size)

for i, model in enumerate(models):

    print(f"Training model: {model.__class__.__name__}")

    model = model.to(device)
    # Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    num_epochs = 10

    # Initialize the current validation loss to infinity
    current_valid_loss = float('inf')

    # initialize the best valid validation accuracy to 0
    best_valid_accuracy = 0.0

    # Initialize the patience counter
    patience_counter = 0

    # Set the maximum number of epochs without improvement
    max_patience = 2

    for epoch in range(num_epochs):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        # Loop over the training data
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / train_size
        train_loss = train_loss / train_size

        model.eval()

        valid_accuracy = 0.0
        valid_loss = 0.0
        # Loop over the validation data
        for i, (images, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            outputs = model(images)
            loss = loss_function(outputs, labels)

            valid_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            valid_accuracy += int(torch.sum(prediction == labels.data))

        valid_accuracy = valid_accuracy / valid_size
        valid_loss = valid_loss / valid_size
        # Print the loss and accuracy for each epoch
        print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Valid Loss: ' + str(valid_loss) + ' Valid Accuracy: ' + str(valid_accuracy))
        # Early stopping
        print('current valid loss: ' + str(current_valid_loss))
        if valid_loss < current_valid_loss:
            current_valid_loss = valid_loss
        else:
            current_valid_loss = valid_loss
            patience_counter += 1
        print('patience counter: ' + str(patience_counter))
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), f'models/best_model_{model.__class__.__name__}.model')
        print('best valid accuracy: ' + str(best_valid_accuracy))
        if patience_counter >= max_patience:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load(f'models/best_model_{model.__class__.__name__}.model'))
