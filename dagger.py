import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import torch
from torchvision import transforms
from PIL import Image
from train_expert import ExpertModel
import os

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Learner(nn.Module):
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

def main():
    env_name = input("What environment should we Test the Expert on  (Tree, Cave)? ")
    env = None

    if(env_name == "Cave"):
        expert_model = ExpertModel(7)
        if(os.path.exists('Expert_model_' + env_name + '.pth')):
            expert_model.load_state_dict(torch.load('Expert_model_' + env_name + '.pth'))
            expert_model.eval()
            env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512)

    else:
        print("Invalid Environment")
        return



if __name__ == "__main__":
    main()