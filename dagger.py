import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import torch
from torchvision import transforms
from PIL import Image
from train_bc import BcModel
from train_bc import ImageDataset
from torch.utils.data import TensorDataset, DataLoader
import os

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Learner(nn.Module):
    def __init__(self, num_actions=0):
        super(Learner, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128) 
        self.fc2 = nn.Linear(128, num_actions) 

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28) 
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# @param env: the environment that the expert model is playing in
# @param learner: the model that we are trying to train
# @param observations: the observations that the expert model has made
# @param actions: the actions that the expert mdeo will take
# @param action_space: the possible actions that can be taken in the enviornemnt
# @param transform: the transformation that we need to apply to the image before giving it to the model  
def interact(env, learner, observations, actions, action_space, transform):
    best_reward = float('-inf')
    for episode in range(20):
        obs, info = env.reset()
        for i in range(40):
            observation = np.array(obs)
            obs = Image.fromarray(observation)
            obs = transform(obs)
            
            _, learner_action = torch.max(learner(obs), 1)
            print("learner_action",learner_action)

            # get the action that the expert would take in this situation
            user_input = input("What action should the agent take type something and press Enter: ")
            if user_input in action_space:
                expert_action = action_space[user_input]
                actions.append(expert_action)
                observations.append(observation)
            
            else:
                print("Invalid action")
    
            
            
            obs, reward, terminated, truncated, _info = env.step(learner_action)
            print(i)
            if terminated:
                break
            
        tensor_action = torch.tensor(actions)
        train(learner, observations, tensor_action, transform)
        reward = evaluate(env, learner, transform)
        print("Reward: ", reward)
        if reward > best_reward:
            best_reward = reward
            torch.save(learner.state_dict(), 'Learner_model.pth')
    
 
# @param learner: the model that we are trying to train
# @param observations: the observations that the expert model (i.e human in this case) has made
# @param actions: the actions that the expert model has taken
# @param transform: the transformation that we need to apply to the image      
def train(learner, observations, actions, transform):
    
    dataset = ImageDataset(observations, actions, transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    learner.train()
    optimizer = optim.Adam(learner.parameters(), lr=0.0001, weight_decay=1e-5)
    for epoch in tqdm(range(100)):
        running_loss = 0.0
        for observation, action in dataloader:
            learner.train()
            learner.zero_grad()
            
            output = learner(observation)
            loss = nn.CrossEntropyLoss()(output, action)
            
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        running_loss /= len(observations)
        print("Epoch: ", epoch, "Loss: ", running_loss)


# @param env: the environment that the expert model is playing in
# @param learner: the model that we are trying to train
# @param transform: the transformation that we need to apply to the image
# @return the total reward that the learner has received
def evaluate(env, learner, transform):
    learner.eval()
    total_reward = 0
    obs, info = env.reset()
    terminated = False
    for j in range(1000):
        obs = np.array(obs)
        obs = Image.fromarray(obs)
        obs = transform(obs)
        with torch.no_grad():
             _, action = torch.max(learner(obs), 1)
        
            
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    return total_reward


def main():
    env_name = input("What environment should we Test the Expert on  (Tree, Cave)? ")
    action_space = dict()
    if(env_name == "Tree"):
        pass
        # learner = Learner(8)
        # expert = ExpertModel(8)
        # if os.path.exists('Expert_model_Tree.pth'):
        #     expert.load_state_dict(torch.load('Expert_model_Tree.pth', weights_only=True))
        #     expert.eval()
        #     env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 512, obs_height = 512)
        # else:
        #     print("No model found")
        #     return

        # env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 512, obs_height = 512)

        # #do nothing
        # action_space['nop'] = 0
        # #forward
        # action_space['f'] = 1
        # #jump
        # action_space['j'] = 2
        # #dig
        # action_space['d'] = 3

        # #mouse controls
        # action_space['+x'] = 4
        # action_space['-x'] = 5
        # action_space['+y'] = 6
        # action_space['-y'] = 7


    elif(env_name == "Cave"):
       
        learner = Learner(7)
        

        #used for multiple test runs
        if os.path.exists('Great_model.pth'):
            learner.load_state_dict(torch.load('Great_model.pth', weights_only=True))
            learner.eval()
        
        #print("Model Loaded")
        env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512, frameskip=2)
        env.mouse_mov = .25

        action_space['nop'] = 0
        action_space['f'] = 1
        #jump
        action_space['j'] = 2

        #mouse controls
        action_space['+x'] = 3
        action_space['-x'] = 4
        action_space['+y'] = 5
        action_space['-y'] = 6

    else:
        print("Invalid Environment")
        return
    
    observations = []
    actions = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    #how we need to transform the image before giving it to the model

    interact(env, learner, observations, actions, action_space, transform)





if __name__ == "__main__":
    main()