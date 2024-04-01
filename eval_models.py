import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prettytable import PrettyTable


# Parameters
current_dir = os.getcwd()
test_dataset_path = os.path.join(current_dir,  "Data/test")
main_model = os.path.join(current_dir, "models/best_model_Main_model.model")
variant_1 = os.path.join(current_dir, "models/best_model_Variant1.model")
variant_2 = os.path.join(current_dir, "models/best_model_Variant2.model")


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()

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


def eval_model(model_path, test_dataset, test_loader):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    state_dict = torch.load(model_path)

    model = ConvNet(num_classes=4)

    # Load the model's state dictionary
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Perform inference on test dataset
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Get class names from the dataset
    class_names = test_dataset.classes

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate precision
    precision_macro = precision_score(true_labels, predicted_labels, average='macro')
    precision_micro = precision_score(true_labels, predicted_labels, average='micro')

    # Calculate recall
    recall_macro = recall_score(true_labels, predicted_labels, average='macro')
    recall_micro = recall_score(true_labels, predicted_labels, average='micro')

    # Calculate F1-measure
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')

    # Print everything
    print("Model: ", model_path.split('/')[-1])

    # Create a PrettyTable instance
    table = PrettyTable()

    # Add column names (predicted labels)
    table.field_names = [""] + [f"{class_name}" for class_name in class_names]

    # Add rows to the table
    for i, class_name in enumerate(class_names):
        row = [f"Actual {class_name}"]
        for j in range(len(class_names)):
            count = conf_matrix[i, j]
            row.append(count)
        table.add_row(row)

    print(table)

    # Create a PrettyTable instance
    table = PrettyTable()

    # Add column names
    table.field_names = ["Metric", "Macro", "Micro"]

    # Add rows to the table
    table.add_row(["Accuracy", "-", f"{accuracy:.4f}"])
    table.add_row(["Precision", f"{precision_macro:.4f}", f"{precision_micro:.4f}"])
    table.add_row(["Recall", f"{recall_macro:.4f}", f"{recall_micro:.4f}"])
    table.add_row(["F1-score", f"{f1_macro:.4f}", f"{f1_micro:.4f}"])

    # Set alignment of columns
    table.align["Metric"] = "l"
    table.align["Macro"] = "r"
    table.align["Micro"] = "r"

    # Print the table
    print(table)
    print()


# Define transform for test images
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    # Convert grayscale to RGB
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create test dataset using ImageFolder
test_dataset = ImageFolder(root=test_dataset_path, transform=transform)

# Define DataLoader for test dataset
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

eval_model(main_model, test_dataset, test_loader)
eval_model(variant_1, test_dataset, test_loader)
eval_model(variant_2, test_dataset, test_loader)


