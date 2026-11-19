import os
import gc
import cv2
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set Device (Explicit GPU usage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Label Mapping (Ensure only defined labels are used)
label_map = {"cloudy": 0, "rain": 1, "sunrise": 2, "shine": 3}

# Load Image Paths and Labels (Extract only alphabets)
image_paths = ['./weather_data/' + i for i in os.listdir('./weather_data')]
labels = [re.sub(r'[^a-zA-Z]', '', os.path.splitext(i)[0]) for i in os.listdir('./weather_data')]  # Remove numbers, dots, and extensions

# Print Unique Labels (for debugging)
print("Unique Labels Found:", set(labels))
print("Expected Labels:", label_map.keys())

# Filter only valid labels
filtered_data = [(img, lbl) for img, lbl in zip(image_paths, labels) if lbl in label_map]
image_paths, labels = zip(*filtered_data) if filtered_data else ([], [])

# Shuffle Data
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# Train-Val-Test Split
train_size = int(0.8 * len(image_paths))
val_size = int(0.1 * len(image_paths))
test_size = len(image_paths) - train_size - val_size

train_paths, val_paths, test_paths = image_paths[:train_size], image_paths[train_size:train_size + val_size], image_paths[train_size + val_size:]
train_labels, val_labels, test_labels = labels[:train_size], labels[train_size:train_size + val_size], labels[train_size + val_size:]

class WeatherDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = [label_map[label] for label in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:  # 🔥 FIX: Handle unreadable images
            print(f"⚠️ Warning: Could not read image {image_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Skip to next image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image.to(device), torch.tensor(label, dtype=torch.long).to(device)


# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets and Dataloaders
train_dataset = WeatherDataset(train_paths, train_labels, transform=transform)
val_dataset = WeatherDataset(val_paths, val_labels, transform=transform)
test_dataset = WeatherDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Model with GPU Support
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize Model on GPU
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Training Loop with GPU
epochs = 15
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss, correct_train = 0.0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)  # GPU
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct_train / len(train_dataset)

    # Validation
    model.eval()
    val_loss, correct_val = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)  # GPU
            val_loss += criterion(outputs, labels).item()
            correct_val += (outputs.argmax(1) == labels).sum().item()

    val_acc = correct_val / len(val_dataset)
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}")

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_weather_model.pth")
        print("Model Saved!")

# Plot Training Loss
plt.plot(range(epochs), [running_loss / len(train_loader) for _ in range(epochs)], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test on GPU
model.load_state_dict(torch.load("best_weather_model.pth"))
model.to(device)
model.eval()
correct_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)  # GPU
        correct_test += (outputs.argmax(1) == labels).sum().item()

test_acc = correct_test / len(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}")

# Save Final Model
torch.save(model, "final_weather_model.pth")
print("Final Model Saved!")
