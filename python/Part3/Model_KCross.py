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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((192, 192)), ## Resizing the Image to 192 x 192 pixels, this is already that size but just to be sure
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder('../../Data_Part3/New', transform=transformer)

# Categories
root = pathlib.Path('../../Data_Part3/New')
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


# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)
best_score = -np.inf

print ('---------------------------------------------------------------')

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=32, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=32, sampler=test_subsampler)

    # Init the neural network
    model = Main(num_classes=4)
    model = model.to(device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    num_epochs = 20


    # Training loop for num_epochs
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
    

    # Print about testing
    print('Starting testing')
    model.eval()
    # Testing process
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():  # Temporarily turn off gradient descent
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy for fold {fold}: {accuracy}%')

    # Calculate precision, recall, f1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    print(f'Macro Precision: {precision}')
    print(f'Macro Recall: {recall}')
    print(f'Macro F1 score: {f1_score}')
    # Save the best model 
    if f1_score > best_score:
        print(f'best model is now {best_score}')
        best_score = f1_score
        print('changing')
        print(f'best model is now {best_score}')
        torch.save(model.state_dict(), f'best_model_PartIII.model')


    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='micro')
    print(f'Micro Precision: {precision}')
    print(f'Micro Recall: {recall}')
    print(f'Micro F1 score: {f1_score}')
    # Process is complete.
    print('Training process has finished. Saving trained model.')
    # Print fold results
    print(f'Accuracy for fold {fold}: {accuracy}%')
    print('--------------------------------')

# Print overall results
print('--------------------------------')
print('K-fold cross validation Done')
print('--------------------------------')
