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


##Sams model type as the expert model
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

#@param expert_model: the expert model that we are trying to mimic
# @param env: the environment that the expert model is playing in
# @param observations: the observations that the expert model has made
# @param learner: the model that we are trying to train
# @param transform: the transformation that we need to apply to the image before giving it to the model  
def interact(expert_model, env, observations, actions, learner, transform):
    best_reward = -100
    for episode in range(5):
        terminated = False
        observation, info = env.reset()
        for i in range(10):
            
            #we need to preprocess the image before giving it to the neural network
            observation = np.array(observation, dtype=np.uint8)
            observation = Image.fromarray(observation)

            observation = transform(observation)
            _, expert_action = torch.max(expert_model(observation), 1)

            _, learner_action = torch.max(learner(observation), 1)

            observation, reward, terminated, truncated, _info = env.step(expert_action)


            observations.append(observation)
            actions.append(expert_action)

            observations.append(observation)
            actions.append(learner_action)

            if terminated:
                break

        
        


        
def train(learner, observations, actions):
    optimizer = optim.Adam(learner.parameters(), lr=0.0001, weight_decay=1e-5)
    for epoch in tqdm(range(100)):
        running_loss = 0.0
        for observations, actions in zip(observations, actions):
            learner.train()
            learner.zero_grad()
            
            output = learner(observations)
            loss = nn.CrossEntropyLoss()(output, actions)
            
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        running_loss /= len(observations)
        print("Epoch: ", epoch, "Loss: ", running_loss)





def main():
    env_name = input("What environment should we Test the Expert on  (Tree, Cave)? ")
    env = None

    if(env_name == "Cave"):
        expert_model = ExpertModel(7)
        if(os.path.exists('Expert_model_' + env_name + '.pth')):
            expert_model.load_state_dict(torch.load('Expert_model_' + env_name + '.pth'))
            expert_model.eval()
            learner = Learner(7)
            env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512)

    else:
        print("Invalid Environment")
        return
    
    observations = []
    actions = []

    #how we need to transform the image before giving it to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    interact(expert_model, env, learner, observations, actions)





if __name__ == "__main__":
    main()