import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
import numpy as np
import os
import h5py


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
    def __init__(self, num_actions=0):
        super(ExpertModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # 128 channels, 28x28 after pooling
        self.fc2 = nn.Linear(128, num_actions)  #number of possible actions

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten for fully connected layer
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(env_type):
    observations = None
    actions = None
    if env_type == "Tree":
        pass
    elif env_type == "Cave":
        #The caves are all the same just different dataruns
        print("Loading expert ...")
        # with h5py.File('expert_data/Cave1.h5', 'r') as f:
        #     observations1 = f['states'][:]
        #     actions1 = f['actions'][:]
        # with h5py.File('expert_data/Cave2.h5', 'r') as f:
        #     observations2 = f['states'][:]
        #     actions2 = f['actions'][:]
        # with h5py.File('expert_data/Cave3.h5', 'r') as f:
        #     observations3 = f['states'][:]
        #     actions3 = f['actions'][:]
        with h5py.File('expert_data/ExpertFinal.h5', 'r') as f:
            observations = f['states'][:]
            actions = f['actions'][:]

        

        #observations = np.concatenate((observations1, observations2, observations3, observations4), axis=0)
        #actions = np.concatenate((actions1, actions2, actions3, actions4), axis=0)
         
       
    else:
        print("Invalid environment type")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ExpertDataset(observations, actions, transform)

    #shuffle the data and split into training and validation sets
    print("Making train test splits ...")
    indices = torch.randperm(len(dataset)).tolist()
    shuffled_data = [dataset[i] for i in indices]
    train_size = int(0.75 * len(shuffled_data))


    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    print("Done making data ...")

    return train_loader, val_loader


def init_model(name):
    num_actions = 0
    if name == "Tree":
        num_actions = 8
    elif name == "Cave":
        num_actions = 7
    else:
        print("Invalid environment type")
        return

    print("initalizing model")
    model = ExpertModel(num_actions)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    return model, criterion, optimizer




def train(env_type, model, criterion, optimizer, train_loader):
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct = 0
        total = 0

        
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


        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        torch.save(model.state_dict(), 'Expert_model_' + env_type + '.pth')
    return model
    

def eval(model, criterion, val_loader):
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        running_loss = running_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Validation Loss: {running_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


def main():
    env = input("What environment should we train for (Tree, Cave)? ")
    
    train_loader, val_loader = load_data(env)
    model, criterion, optimizer = init_model(env)
    # if(os.path.exists('Expert_model_' + env + '.pth')):
    #     #model.load_state_dict(torch.load('Expert_model_' + env + '.pth'))
    #     pass
    
    model = train(env, model, criterion, optimizer, train_loader)
    
    eval(model, criterion, val_loader)

    

if __name__ == "__main__":
    main()

