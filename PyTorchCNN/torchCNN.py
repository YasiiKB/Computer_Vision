# Import dependencies
from PIL import Image
import torch
import torchvision
from torch import nn, save, load
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import DataLoader 
from torchvision.datasets import MNIST

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # we'll send deta to device

# Define a simple CNN model
'''
Layer Sizes: 

kernel_size = number * number
conv1_output_size = input_size - (kernel_size - 1)
conv2_output_size = conv1_output_size - (kernel_size - 1)
conv3_output_size = conv2_output_size - (kernel_size - 1)
Linear_output_layer_size = last conv_size * (image_size - (kernel_size - 1)* number of conv layers) * that again * output classes 
Here (image_size - (kernel_size - 1)* number of conv layers) = (28 - (3-1)* 3)
'''
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3), # input feature = 1 becasuse MNIST is in grayscale
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear (64*(28-6)*(28-6), 10)  # 10 output features :  10 classes in the MNIST dataset (digits 0 through 9).
).to(device)

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean=0.1307, std=0.3081
])

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


# Train and Test set
train_dataset = MNIST(root='./PyTorchCNN/data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./PyTorchCNN/data', train=False, download=True, transform=transform)

# Shuffle 
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Apply backprop 
        optimizer.zero_grad() # gradient descent 
        outputs = model(images)
        loss = loss_func(outputs, labels) 
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
torch.save(model.state_dict(), './PyTorchCNN/model_state.pt')

# Load model 
# model = torch.load('./PyTorchCNN/model_state.pt')

# Test on a single image
img = Image.open('./PyTorchCNN/img_3.jpg') 
img_tensor = ToTensor()(img).unsqueeze(0).to(device)

print(torch.argmax(model(img_tensor)))