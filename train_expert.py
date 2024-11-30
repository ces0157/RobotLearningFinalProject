import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
import numpy as np


class ExpertDataset(Dataset):
    def __init__(self, observations, actions, transform = None):
        self.observations = observations
        self.actions = actions
        self.transform = transform

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, index):
        state = np.array(self.observations[index])
        state = state.astype(np.uint8)

        state = Image.fromarray(state)
        action = self.actions[index]

        if self.transform:
            state = self.transform(state)
        
        return state, action


class ExpertModel(nn.Module):
    def __init__(self):
        super(ExpertModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # 128 channels, 28x28 after pooling
        self.fc2 = nn.Linear(128, 8)  # 7 possible actions

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten for fully connected layer
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x





def train():
    with open("data/Tree.bin", "rb") as binary_file:
        data = pickle.load(binary_file)
    
    print("Loading observations...")
    observations = data["state"]
    print("Loading actions...")
    actions = data["action"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Making dataset...")

    dataset = ExpertDataset(observations, actions, transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for input, labels in train_loader:
    #     print("Hello there")

    
    print("initalizing model")
    model = ExpertModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, unit='batch', desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (total / 32), accuracy=correct / total)


        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        torch.save(model.state_dict(), 'Expert_model.pth')

    


train()