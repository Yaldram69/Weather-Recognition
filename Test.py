import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Label Mapping (Ensure the order is the same as training)
class_labels = ["cloudy", "rain", "sunrise", "shine"]

# Define the Same CNN Model as Training
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


# Load Trained Model
model = torch.load("final_weather_model.pth", map_location=device)  # Load full model
model.to(device)
model.eval()

# Define Image Transformations (Same as Training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Function to Predict Weather from Image
def predict_weather(image_path):
    # Load Image
    image = cv2.imread(image_path)

    if image is None:  # Handle unreadable images
        print(f"⚠️ Error: Could not read image {image_path}")
        return

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    image_pil = Image.fromarray(image)

    # Apply Transformations
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_label = class_labels[predicted.item()]
    print(f"Predicted Weather Condition: {predicted_label}")

    # Display Image with Prediction
    cv2.putText(image, f"Predicted: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test the Model with an Image
test_image_path = r"C:\Users\CREATIVE IT PARK\Downloads\aa.jpg"  # Change to your test image path
predict_weather(test_image_path)
