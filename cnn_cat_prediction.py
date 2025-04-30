
import os

for files in os.listdir(cat_test_path):
  print(files)

"""Data preprocessing"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image, UnidentifiedImageError

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Data preparation
IMG_SIZE = 32 #32x32 pixel
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), #resize image
    transforms.ToTensor(), #convert image to tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #normalize pixel from -1 to 1
])

class FilteredImageFolder(Dataset):
    def __init__(self, root, max_samples_per_class, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._filter_images(max_samples_per_class)

    def _filter_images(self, max_samples_per_class):
      samples = []
      class_counts = {}

      for class_idx, class_name in enumerate(sorted(os.listdir(self.root))):
          class_folder = os.path.join(self.root, class_name)

          if os.path.isdir(class_folder):

              class_counts[class_name] = 0

              for filename in os.listdir(class_folder):

                  if class_counts[class_name] >= max_samples_per_class:
                    break

                  filepath = os.path.join(class_folder, filename)
                  try:
                      with Image.open(filepath) as img:
                          #img.verify()

                          img = img.convert("RGB")
                          img.load()

                      samples.append((filepath, class_idx))
                      class_counts[class_name] += 1
                      print(class_counts)
                      #if count >= max_samples_per_class:
                        #return samples

                  except UnidentifiedImageError:
                      print(f"UnidentifiedImageError: Skipping file {filepath}")
                  except IOError:
                      print(f"IOError: Skipping file {filepath}")
                  except Exception as e:
                      print(f"Unexpected error {e}: Skipping file {filepath}")
      return samples

    def __len__(self):
        return len(self.samples)

    #updated version
    def __getitem__(self, idx):
      filepath, label = self.samples[idx]
      try:
          with Image.open(filepath) as img:
              img = img.convert("RGB")  # Ensure it's a 3-channel image
              if self.transform:
                  img = self.transform(img)
          return img, label
      except Exception as e:
          print(f"Error loading {filepath}: {e}")
          # Return a dummy tensor and an invalid label (-1), or retry another index
          return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

# Initialize datasets with filtered images
train_dataset = FilteredImageFolder(train_path, 750, transform=transform)
test_dataset = FilteredImageFolder(test_path, 250, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model initialization
model = models.resnet18(pretrained=True)  # Example model
model.to(device)

# Load the model if it exists
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
else:
    # Train your model here if not already trained
    print("Training the model...")

    # Example training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # Example epoch count
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(train_dataset.samples)  # Print samples to verify

print(test_dataset.samples)

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def visualize_sample_images(image_folder, num_samples=5):
    class_names = os.listdir(image_folder)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, class_name in enumerate(class_names):
        class_folder = os.path.join(image_folder, class_name)
        sample_files = os.listdir(class_folder)[:num_samples]
        for j, file in enumerate(sample_files):
            img_path = os.path.join(class_folder, file)
            img = Image.open(img_path)
            axes[j].imshow(img)
            axes[j].set_title(class_name)
            axes[j].axis('off')
    plt.show()

def plot_image_histogram(image_folder, num_samples=5):
    class_names = os.listdir(image_folder)
    for class_name in class_names:
        class_folder = os.path.join(image_folder, class_name)
        sample_files = os.listdir(class_folder)[:num_samples]
        for file in sample_files:
            img_path = os.path.join(class_folder, file)
            img = Image.open(img_path)
            img_array = np.array(img)
            plt.figure(figsize=(8, 4))
            plt.hist(img_array.ravel(), bins=256, color='orange', alpha=0.7, rwidth=0.85)
            plt.title(f'Pixel Intensity Distribution - {class_name}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

# Example usage:
visualize_sample_images(train_path)
plot_image_histogram(train_path)

"""Training the cnn"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# 1. Data preparation with augmentation
IMG_SIZE = 32
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class FilteredImageFolder(Dataset):
    def __init__(self, root, max_samples_per_class, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._filter_images(max_samples_per_class)

    def _filter_images(self, max_samples_per_class):
        samples = []
        class_counts = {}

        for class_idx, class_name in enumerate(sorted(os.listdir(self.root))):
            class_folder = os.path.join(self.root, class_name)
            if os.path.isdir(class_folder):
                class_counts[class_name] = 0
                for filename in os.listdir(class_folder):
                    if class_counts[class_name] >= max_samples_per_class:
                        break
                    filepath = os.path.join(class_folder, filename)
                    try:
                        with Image.open(filepath) as img:
                            img = img.convert("RGB")
                            img.load()
                        samples.append((filepath, class_idx))
                        class_counts[class_name] += 1
                    except Exception as e:
                        print(f"Skipping file {filepath}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

# Initialize datasets with augmented images
train_dataset = FilteredImageFolder(train_path, 750, transform=transform)
test_dataset = FilteredImageFolder(test_path, 250, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 2. Define a CNN with Dropout
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = CatDogCNN().to(device)

# 3. Loss and optimizer with weight decay
criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Adam optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # sgd optimizer - 68
optimizer = optim.RMSprop(model.parameters(), lr=0.001) #rmsprop - 73/71.2
# optimizer = optim.AdamW(model.parameters(), lr=0.001) #adamw - 70/68

# 4. Train the model
EPOCHS = 20
best_val_loss = float('inf')
early_stop_count = 0
early_stop_patience = 3  # Stop if validation loss doesn't improve for 3 epochs

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Evaluate on validation set for early stopping
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            print("Early stopping triggered.")
            break

# Final evaluation on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.4f}")

"""Evaluation metric and visualization"""

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Function to calculate precision, recall, F1 score, and AUC
def calculate_metrics(true_labels, predicted_labels):
    true_positive = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positive = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negative = np.sum((true_labels == 1) & (predicted_labels == 0))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Evaluate on the entire test set for more stable metrics
all_labels = []
all_preds = []
all_outputs = []  # Collect raw outputs for AUC calculation
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        all_labels.extend(labels.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_outputs.extend(outputs.cpu().numpy().flatten())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
    auc = roc_auc_score(np.array(all_labels), np.array(all_outputs))

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(np.array(all_labels), np.array(all_outputs))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

"""feature maps"""

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

def visualize_feature_maps(model, image_tensor, device):
    model.eval()
    # Access the first convolutional layer
    layer = model.model[0]
    with torch.no_grad():
        feature_maps = layer(image_tensor.unsqueeze(0).to(device))
        feature_maps = feature_maps.squeeze(0).cpu().numpy()

    num_feature_maps = feature_maps.shape[0]
    # Limit the number of feature maps to plot for clarity
    num_to_plot = min(num_feature_maps, 10)
    fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 15))
    for i in range(num_to_plot):
        axes[i].imshow(feature_maps[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

def visualize_dataset_feature_maps(dataset, model, device, num_images=5):
    for i in range(num_images):
        image, _ = dataset[i]  # Get image and label
        visualize_feature_maps(model, image, device)

# Example usage:
# Visualize feature maps for a subset of images from train and test datasets
visualize_dataset_feature_maps(train_dataset, model, device, num_images=5)
visualize_dataset_feature_maps(test_dataset, model, device, num_images=5)

"""alternate execution : densenet implementation"""

import torch.nn as nn
from torchvision import models

class DenseNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(DenseNetBinaryClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet(x)

model = DenseNetBinaryClassifier().to(device)

"""training the model"""

import torch
import torch.optim as optim

# Initialize the model, criterion, and optimizer
model = DenseNetBinaryClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.4f}")


# # Training loop with early stopping
# EPOCHS = 20
# best_val_loss = float('inf')
# early_stop_count = 0
# early_stop_patience = 3  # Stop if validation loss doesn't improve for 3 epochs

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0
#     correct = 0
#     total = 0

#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         preds = (outputs > 0.5).float()
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     train_accuracy = correct / total
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.4f}")

#     # Evaluate on validation set for early stopping
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:  # Use a separate validation set if possible
#             inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             preds = (outputs > 0.5).float()
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     val_accuracy = correct / total
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         early_stop_count = 0
#     else:
#         early_stop_count += 1
#         if early_stop_count >= early_stop_patience:
#             print("Early stopping triggered.")
#             break

# # Final evaluation on test set
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
#         outputs = model(inputs)
#         preds = (outputs > 0.5).float()
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

# print(f"Test Accuracy: {correct/total:.4f}")

"""testing"""

# Evaluate on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

"""evaluation metrics and visualization"""

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Function to calculate precision, recall, F1 score, and AUC
def calculate_metrics(true_labels, predicted_labels):
    true_positive = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positive = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negative = np.sum((true_labels == 1) & (predicted_labels == 0))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Evaluate on the entire test set for more stable metrics
all_labels = []
all_preds = []
all_outputs = []  # Collect raw outputs for AUC calculation
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        all_labels.extend(labels.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_outputs.extend(outputs.cpu().numpy().flatten())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
    auc = roc_auc_score(np.array(all_labels), np.array(all_outputs))

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(np.array(all_labels), np.array(all_outputs))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

"""feature maps visualization"""

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

def visualize_feature_maps(model, image_tensor, device):
    model.eval()
    # Access the first convolutional layer in DenseNet
    layer = model.densenet.features[0]  # Accessing the first convolutional layer
    with torch.no_grad():
        feature_maps = layer(image_tensor.unsqueeze(0).to(device))
        feature_maps = feature_maps.squeeze(0).cpu().numpy()

    num_feature_maps = feature_maps.shape[0]
    # Limit the number of feature maps to plot for clarity
    num_to_plot = min(num_feature_maps, 10)
    fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 15))
    for i in range(num_to_plot):
        axes[i].imshow(feature_maps[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

def visualize_dataset_feature_maps(dataset, model, device, num_images=5):
    for i in range(num_images):
        image, _ = dataset[i]  # Get image and label
        visualize_feature_maps(model, image, device)

# Example usage:
# Visualize feature maps for a subset of images from train and test datasets
visualize_dataset_feature_maps(train_dataset, model, device, num_images=5)
visualize_dataset_feature_maps(test_dataset, model, device, num_images=5)