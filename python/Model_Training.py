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
import matplotlib
from torchvision import datasets
from Architectures import Main, Variant1, Variant2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#transforms
transformer = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Data Being split
dataset = datasets.ImageFolder('../Data/train', transform=transformer)
train_size = int(0.7 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])


#DataLoaders 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

#categories 
root = pathlib.Path('../Data/train')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

models = [Main(num_classes=4), Variant1(num_classes=4), Variant2(num_classes=4)]


print(train_size, valid_size, test_size)



for i, model in enumerate(models):
    
    print(f"Training model: {model.__class__.__name__}")

    
    model = model.to(device)
    # Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    num_epochs = 2

    # Initialize the best validation loss to infinity
    best_valid_loss = float('inf')

    # Initialize the patience counter
    patience_counter = 0

    # Set the maximum number of epochs without improvement
    max_patience = 2

    for epoch in range(num_epochs):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.cpu().data*images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_accuracy += int(torch.sum(prediction == labels.data))
            
        train_accuracy = train_accuracy / train_size
        train_loss = train_loss / train_size
        
        model.eval()
        
        valid_accuracy = 0.0
        valid_loss = 0.0
        
        for i, (images, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            valid_loss += loss.cpu().data*images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            valid_accuracy += int(torch.sum(prediction == labels.data))
            
        valid_accuracy = valid_accuracy / valid_size
        valid_loss = valid_loss / valid_size
        
        print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Valid Loss: '+str(valid_loss)+' Valid Accuracy: '+str(valid_accuracy))
        
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f'models/best_model_{model.__class__.__name__}.model')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load(f'models/best_model_{model.__class__.__name__}.model'))
