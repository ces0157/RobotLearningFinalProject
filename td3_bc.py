import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer1(state))
        a = torch.relu(self.layer2(a))
        return self.max_action * torch.tanh(self.layer3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.layer1(torch.cat([state, action], 1)))
        q = torch.relu(self.layer2(q))
        return self.layer3(q)

class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, alpha=2.5):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic1 = Critic(state_dim, action_dim).cuda()
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
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, expert_data_file, batch_size=256):
        # Load expert data from the H5 file
        with h5py.File(expert_data_file, 'r') as f:
            states = np.array(f['states'])
            actions = np.array(f['actions'])
            next_states = np.array(f['next_states'])
            rewards = np.array(f['reward'])
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        rewards = torch.FloatTensor(rewards).cuda()

        # Training loop
        for i in range(len(states) // batch_size):
            # Sample a batch of data
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_states = states[batch_start:batch_end]
            batch_actions = actions[batch_start:batch_end]
            batch_next_states = next_states[batch_start:batch_end]
            batch_rewards = rewards[batch_start:batch_end]
            
            # Add noise to the next actions for exploration (TD3 part)
            noise = (torch.randn_like(batch_actions) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(batch_next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic1_target(batch_next_states, next_action)
            target_Q2 = self.critic2_target(batch_next_states, next_action)
            target_Q = batch_rewards + (1 - 0) * self.discount * torch.min(target_Q1, target_Q2)

            # Get current Q estimates
            current_Q1 = self.critic1(batch_states, batch_actions)
            current_Q2 = self.critic2(batch_states, batch_actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            current_action = self.actor(batch_states)
            actor_loss = -self.critic1(batch_states, current_action).mean() + self.alpha * nn.MSELoss()(current_action, batch_actions)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)