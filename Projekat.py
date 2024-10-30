import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Load data
X_train = np.loadtxt('input.csv', delimiter=',').reshape(-1, 100, 100, 3).astype(np.float32) / 255.0
Y_train = np.loadtxt('labels.csv', delimiter=',').reshape(-1, 1).astype(np.float32)
X_test = np.loadtxt('input_test.csv', delimiter=',').reshape(-1, 100, 100, 3).astype(np.float32) / 255.0
Y_test = np.loadtxt('labels_test.csv', delimiter=',').reshape(-1, 1).astype(np.float32)



# Convert to tensors
X_train_tensor = torch.tensor(X_train).permute(0, 3, 1, 2)
Y_train_tensor = torch.tensor(Y_train)
X_test_tensor = torch.tensor(X_test).permute(0, 3, 1, 2)
Y_test_tensor = torch.tensor(Y_test)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=64, shuffle=False)

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate model, define loss and optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on test data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Prediction on a single random image from test set
idx2 = random.randint(0, len(Y_test) - 1)
plt.imshow(X_test[idx2])
plt.show()

with torch.no_grad():
    image = X_test_tensor[idx2].unsqueeze(0)
    output = model(image)
    prediction = 'pas' if output < 0.5 else 'macka'
    print("Na slici se nalazi:", prediction)
