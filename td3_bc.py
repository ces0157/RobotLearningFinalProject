import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class TD3Dataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations ,transform = None):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.transform = transform

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, index):
        state = self.observations[index]
        state = Image.fromarray(state)
        state = state.resize((64, 64))
        state = np.array(state)
        #state /= 255
        state = state.flatten()
        state = torch.tensor(state, dtype=torch.float32)
        state = state.cuda()
        
        
        action = self.actions[index]
        action = torch.tensor(action, dtype=torch.long)
        action = F.one_hot(action,  num_classes=7)
        action = action.float()
        action = action.cuda()
        
        reward = self.rewards[index]
        reward = torch.tensor(reward, dtype = torch.float32)
        reward = reward.cuda()

        next_state = self.next_observations[index]
        next_state = Image.fromarray(next_state)
        next_state = next_state.resize((64, 64))
        next_state = np.array(next_state)
        #next_state /= 255
        next_state = next_state.flatten()
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state = next_state.cuda()

        # if self.transform:
        #     state = self.transform(state)
        #     next_state = self.transform(next_state)
        
        return state, action, reward, next_state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)

        # self.layer1 = nn.Linear(128 * 28 * 28, 128)
        # self.layer2 = nn.Linear(128, action_dim)
        # #self.layer3 = nn.Linear(128, action_dim)
        
        # # self.layer1 = nn.Linear(state_dim, 256)
        # # self.layer2 = nn.Linear(256, 256)
        # # self.layer3 = nn.Linear(256, action_dim)
        
        
        #self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer1(state))
        a = torch.relu(self.layer2(a))
        a = self.layer3(a)
        return a
        
        #x = self.pool(nn.ReLU()(self.conv1(state)))
        # x = self.pool(nn.ReLU()(self.conv2(x)))
        # x = self.pool(nn.ReLU()(self.conv3(x)))
        # x = x.view(-1, 128 * 28 * 28)
        # x = torch.relu(self.layer1(x))
        # x= self.layer2(x)
        # return x
        
        
        
        # a = torch.relu(self.layer1(state))
        # a = torch.relu(self.layer2(a))
        # return self.max_action * torch.tanh(self.layer3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        


       

    def forward(self, state_action):
        #action = action.view(-1,1)
        q = torch.relu(self.layer1(state_action))
        q = torch.relu(self.layer2(q))
        return self.layer3(q)
        
        
        # q = self.pool(nn.ReLU()(self.conv1(state)))
        # q = self.pool(nn.ReLU()(self.conv2(q)))
        # q = self.pool(nn.ReLU()(self.conv3(q)))
        # #q = self.pool(nn.ReLU()(self.conv4(q)))

        # q = q.view(-1, 128 * 28 * 28)
        # q = torch.relu(self.layer1(q))
        # action = action.view(-1, 1)

        # q = torch.relu(self.layer2(torch.cat([q, action ], 1)))
        # q = self.layer3(q)
        # #q = self.layer4(q)
        # return q

class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, alpha=2.5):
        self.actor = Actor(state_dim,action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic1 = Critic(state_dim,action_dim).cuda()
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim, action_dim).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, dataloader, epochs = 50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Convert data to PyTorch tensors
        #states = torch.FloatTensor(states).cuda()
        #actions = torch.FloatTensor(actions).cuda()
        #next_states = torch.FloatTensor(next_states).cuda()
        #rewards = torch.FloatTensor(rewards).cuda()

        # Training loop
        #for epoch in tqdm(range(len(states) // batch_size)):
        for epoch in tqdm(range(epochs)):
            print("Epoch: ", epoch)
            for batch_states, batch_actions, batch_rewards, batch_next_states in dataloader:
            # Sample a batch of data
            # batch_start = i * batch_size
            # batch_end = (i + 1) * batch_size
            # batch_states = states[batch_start:batch_end]
            # batch_actions = actions[batch_start:batch_end]
            # batch_next_states = next_states[batch_start:batch_end]
            # batch_rewards = rewards[batch_start:batch_end]
            
                # Add noise to the next actions for exploration (TD3 part)
                # batch_states = batch_states.cuda()
                # batch_actions = batch_actions.cuda()
                # batch_rewards = batch_rewards.cuda()
                # batch_next_states = batch_next_states.cuda()
                
                next_action = self.actor_target(batch_next_states)
                # noise = (torch.randn_like(x) * 0.2).clamp(-0.5, 0.5)
                # next_action = (x + noise).clamp(0, self.max_action)

                #make a one hot encoding
                indices = torch.argmax(next_action, dim=1)
                next_action = torch.eye(7, device=next_action.device)[indices]
                
                state_next_action = torch.cat((batch_next_states, next_action), 1)
                
               
                #next_action = next_action.reshape(7,1)
                #print(next_action)
                #print(next_action.shape)


                # Compute the target Q value
                #print("Computing target Q value")
                target_Q1 = self.critic1_target(state_next_action)
                target_Q2 = self.critic2_target(state_next_action)

                #target_Q1 = self.critic1_target(batch_next_states, next_action)
                #target_Q2 = self.critic2_target(batch_next_states, next_action)
                batch_rewards =  batch_rewards.reshape(batch_rewards.shape[0], 1)
                target_Q = batch_rewards + (1 - 0) * self.discount * torch.min(target_Q1, target_Q2)

                # Get current Q estimates 
                #print("Getting current Q estimates")
                #convert to state_action pair
                #print(batch_states.shape, batch_actions.shape)
                state_action = torch.cat((batch_states, batch_actions), 1)

                current_Q1 = self.critic1(state_action)
                current_Q2 = self.critic2(state_action)


                #current_Q1 = self.critic1(batch_states, batch_actions)
                #current_Q2 = self.critic2(batch_states, batch_actions)

                # Compute critic loss
                critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
                print("Critic Loss: ", critic_loss.item())
                
                # Optimize the critics
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute actor loss
                _, current_action = torch.max(self.actor(batch_states), 1)
                Q_value = self.critic1(state_action)
                lmda = self.alpha / Q_value.abs().mean().detach()


                
                current_action = current_action.float()
                current_action = F.one_hot(current_action.long(),  num_classes=7)
                current_action = current_action.float()

                print(current_action)
                print(batch_actions)
                #batch_actions = batch_actions.argmax(dim=1).float()
                #actor_loss = nn.CrossEntropyLoss()(current_action, batch_actions)
                #actor_loss = -lmda * Q_value.mean() + nn.CrossEntropyLoss()(current_action, batch_actions)
                actor
                actor_loss = -self.critic1(state_action).mean() + self.alpha * nn.CrossEntropyLoss()(current_action, batch_actions)

                
                print("Actor Loss: ", actor_loss.item())
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                #print("Going to update")
                # Update the target networks
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            torch.save(self.actor.state_dict(), 'Actor_model_td3.pth')



def transform_data(expert_data_file, batch_size = 256):
    with h5py.File(expert_data_file, 'r') as f:
        states = np.array(f['states'])
        actions = np.array(f['actions'])
        next_states = np.array(f['next_states'])
        rewards = np.array(f['reward'])
    

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
    #     transforms.ToTensor(),
    #     #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

   
    

    dataset = TD3Dataset(states, actions, rewards, next_states)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
   
    return dataloader


def main():
    
    user_input = input("What environment should we train for the Expert on  (Tree, Cave)? ")
    action_dim = 0
    if user_input == "Tree":
        pass
    elif user_input == "Cave":
        expert_data_file = "expert_data/CaveUpdated.h5"
        action_dim = 7
    else:
        print("Invalid Environment")
        return
    
    print("loading data ...")
    dataloader = transform_data(expert_data_file)
    print("Data loaded")
    print("Initializing TD3_BC ...")
    td3 = TD3_BC(state_dim = 12288 ,action_dim=action_dim, max_action=6)
    print("Training TD3_BC ...")
    td3.train(dataloader)




if __name__ == "__main__":
    main()

