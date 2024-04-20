from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
from torchvision import datasets, transforms
from Architectures import Main
import os


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # Override the __getitem__ method to return the file name along with the image and its label
        path, target = self.samples[index]
        image = self.loader(path)
        
        # Extract the age and gender information from the file name
        filename, _ = os.path.splitext(os.path.basename(path))
        age, gender = filename.split('_')[-2:]
        
        # Convert age and gender strings to integers or categorical variables if necessary
        
        # Apply transformations to the image if required
        if self.transform is not None:
            image = self.transform(image)

        return image, target, age, gender

# Define the path to your root folder
root_folder = '../Data_Part3/root/'

# Open a file in write mode ('w')
file = open('metrics_output.txt', 'w')

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define the dataset using ImageFolder
dataset = CustomImageFolder(root=root_folder, transform=transform)

# Define the DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Print the classes/categories
file.write(str(dataset.classes) + "\n")
# Print the number of samples in the dataset
file.write(str(len(dataset)) + "\n")
# Separator
file.write("------" + "\n")

# Initialize counters for each age and gender category
age_gender_counts = {('young', 'male'): 0, ('young', 'female'): 0, ('young', 'other'): 0,
                     ('middle', 'male'): 0, ('middle', 'female'): 0, ('middle', 'other'): 0,
                     ('senior', 'male'): 0, ('senior', 'female'): 0, ('senior', 'other'): 0}

# Load the saved model
model_path = 'models/best_model_Main.model'
state_dict = torch.load(model_path)
model = Main(num_classes=4)

# Load the model's state dictionary
model.load_state_dict(state_dict)
model.eval()

# Make predictions and obtain ground truth labels
all_predictions = []
all_labels = []
all_ages = []
all_genders = []
for image, label, age, gender in dataset:
    # Add batch dimension to the images tensor
    image = image.unsqueeze(0)  # Add batch dimension at index 0

    # Make predictions using your model
    predictions = model(image)
    _, predicted_labels = torch.max(predictions, 1)

    # Update counters based on predicted labels
    age_gender_counts[(age, gender)] += 1

    all_predictions.extend(predicted_labels.cpu().numpy())
    all_labels.append(label)
    all_ages.append(age)
    all_genders.append(gender)

# Print the counts for each age and gender category
file.write("Counts for each age and gender category:" + "\n")
for key, value in age_gender_counts.items():
    file.write(f"{key}: {value}" + "\n")

# Separator
file.write("------" + "\n")

# Convert age and gender labels to numerical representations (if needed)
age_dict = {'young': 0, 'middle': 1, 'senior': 2}
gender_dict = {'male': 0, 'female': 1, 'other': 2}

# Convert age and gender labels from strings to numerical representations using the dictionaries
all_ages = [age_dict[age] for age in all_ages]
all_genders = [gender_dict[gender] for gender in all_genders]

# Convert lists to numpy arrays for easier manipulation
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_ages = np.array(all_ages)
all_genders = np.array(all_genders)

# Calculate accuracy, precision, recall, and F1-score for each subclass
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average=None)
recall = recall_score(all_labels, all_predictions, average=None)
f1 = f1_score(all_labels, all_predictions, average=None)

# Print the metrics
file.write("\n")
file.write(f"Overall Accuracy: {accuracy}" + "\n")
file.write("\n")
file.write("Precision for each class:" + "\n")
for subclass, precision_scores in zip(dataset.classes, precision):
    file.write(f"{subclass}: {precision_scores}" + "\n")
file.write("\n")
file.write("Recall for each class:" + "\n")
for subclass, recall_scores in zip(dataset.classes, recall):
    file.write(f"{subclass}: {recall_scores}" + "\n")
file.write("\n")
file.write("F1-score for each class:" + "\n")
for subclass, f1_scores in zip(dataset.classes, f1):
    file.write(f"{subclass}: {f1_scores}" + "\n")
file.write("\n")
file.write("\n")
file.write("\n")

# Separator
file.write("------" + "\n")

# Define a function to compute metrics for each subclass
def compute_metrics_for_subclass(isAge, subclass, predictions, labels):
    # Filter predictions and labels for the specified subclass
    subclass_indices = [age == subclass for age in all_ages] if isAge else [gender == subclass for gender in all_genders]
    subclass_predictions = predictions[subclass_indices]
    subclass_labels = labels[subclass_indices]
    
    # Compute metrics
    accuracy = accuracy_score(subclass_labels, subclass_predictions)
    precision = precision_score(subclass_labels, subclass_predictions, average=None, zero_division=0)
    recall = recall_score(subclass_labels, subclass_predictions, average=None, zero_division=0)
    f1 = f1_score(subclass_labels, subclass_predictions, average=None, zero_division=0)
    
    return accuracy, precision, recall, f1

# Get the class names from the dataset
class_names = dataset.classes

# Compute metrics for each age subclass
subclasses = ['young', 'middle', 'senior']
for s_subclass in subclasses:
    file.write(f"Metrics for subclass {s_subclass}:" + "\n")
    subclass = age_dict[s_subclass]
    accuracy, precision, recall, f1 = compute_metrics_for_subclass(True, subclass, all_predictions, all_labels)
    file.write(f"Accuracy: {accuracy}" + "\n")
    for i, class_name in enumerate(class_names):
        if s_subclass == "middle":
            if class_name == "engaged":
                continue

            else:
                i -= 1
                file.write(f"\tClass: {class_name}" + "\n")
                file.write(f"\tPrecision: {precision[i]}" + "\n")
                file.write(f"\tRecall: {recall[i]}" + "\n")
                file.write(f"\tF1-score: {f1[i]}" + "\n")
                file.write("\n")

        else:
            file.write(f"\tClass: {class_name}" + "\n")
            file.write(f"\tPrecision: {precision[i]}" + "\n")
            file.write(f"\tRecall: {recall[i]}" + "\n")
            file.write(f"\tF1-score: {f1[i]}" + "\n")
            file.write("\n")

    file.write("\n")

# Compute metrics for each gender subclass
subclasses = ['male', 'female', 'other']
for subclass in subclasses:
    file.write(f"Metrics for subclass {subclass}:" + "\n")
    subclass = gender_dict[subclass]
    accuracy, precision, recall, f1 = compute_metrics_for_subclass(False, subclass, all_predictions, all_labels)
    file.write(f"Accuracy: {accuracy}" + "\n")
    for i, class_name in enumerate(class_names):
        file.write(f"\tClass: {class_name}" + "\n")
        file.write(f"\tPrecision: {precision[i]}" + "\n")
        file.write(f"\tRecall: {recall[i]}" + "\n")
        file.write(f"\tF1-score: {f1[i]}" + "\n")
        file.write("\n")
    file.write("\n")


# Close the file
file.close()
