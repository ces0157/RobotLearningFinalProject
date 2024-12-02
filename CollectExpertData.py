import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import json
import pickle
import h5py



data = dict()
data['state'] = []
data['next_state'] = []
data['reward'] = []
data['action'] = []
env_name = input("What environment should the agent play in (Tree, Cave)? ")
action_space = dict()


if(env_name == "Tree"):
    env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 640, obs_height = 360)
    observation, info = env.reset()

    #do nothing
    action_space['nop'] = 0
    #forward
    action_space['f'] = 1
    #jump
    action_space['j'] = 2
    #dig
    action_space['d'] = 3

    #mouse controls
    action_space['+x'] = 4
    action_space['-x'] = 5
    action_space['+y'] = 6
    action_space['-y'] = 7

if(env_name == "Cave"):
    env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512)
    observation, info = env.reset()

    action_space['nop'] = 0
    action_space['f'] = 1
    #jump
    action_space['j'] = 2

    #mouse controls
    action_space['+x'] = 3
    action_space['-x'] = 4
    action_space['+y'] = 5
    action_space['-y'] = 6

for t in range(300):
    
    user_input = input("What action should the agent take type something and press Enter: ")
    
    if user_input not in action_space:
        print("Invalid action")
        continue

    action = action_space[user_input]
    
    data['state'].append(observation.tolist())
    data['action'].append(action)
    
    observation, reward, terminated, truncated, _info = env.step(action)
        
    data['next_state'].append(observation.tolist())
    data['reward'].append(reward)

    print(t)
    
    #only used for the Cave environment
    if reward > 24:
       break


env.close()


#after one expert collection, copy the data multiple times to make the expert consistent (at leat for the cave)
for i in range(2):
    print(i)
    data['state'] = np.concatenate((data['state'], data['state']), axis=0)
    data['action'] = np.concatenate((data['action'], data['action']), axis=0)
    data['next_state'] = np.concatenate((data['next_state'], data['next_state']), axis=0)
    data['reward'] = np.concatenate((data['reward'], data['reward']), axis=0)


print("Saving data")
with h5py.File('expert_data/ExpertFinal2.h5', 'w') as f:
    f.create_dataset('states', data= data['state'], compression='gzip', compression_opts=9)
    f.create_dataset('actions', data=data['action'])
    f.create_dataset('next_states', data=data['next_state'], compression='gzip', compression_opts=9)
    f.create_dataset('reward', data=data['reward'])
   
    






