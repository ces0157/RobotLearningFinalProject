import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import json
import pickle
import h5py
from PIL import Image
import time
from craftium.wrappers import BinaryActionWrapper


data = dict()
data['state'] = []
data['next_state'] = []
data['reward'] = []
data['action'] = []
env_name = input("What environment should the agent play in (Tree, Cave): ")
action_space = dict()


craftium_kwargs = dict(
            frame_skip = 3,
            rgb_observations=True,
            gray_scale_keepdim=False,
        )


if(env_name == "Tree"):
    
    
    env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 512, obs_height = 512, frameskip=15)
    # env = BinaryActionWrapper(env, 
    #                           actions= ["forward", "jump", "dig", "mouse x+", "mouse x-", "mouse y+", "mouse y-"],
    #                           mouse_mov=.25,)
    #env.mouse_mov = .05
    print(env.mouse_mov)
    observation, info = env.reset()

    print(env.action_space)

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
    
    env = gym.make("Craftium/Speleo-v0", render_mode = "human",  obs_width = 512, obs_height = 512, frameskip=2)
    env.mouse_mov = .25
    
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




observation = np.array(observation)
for t in range(500):

    user_input = input("What action should the agent take type something and press Enter: ")
    if user_input not in action_space:
        print("Invalid action")
        continue
    
    action = action_space[user_input]
    print(action)
    data['action'].append(action)
    data['state'].append(observation)

    
    observation, reward, terminated, truncated, _info = env.step(action)
    observation = np.array(observation)

    observation = np.array(observation)
    data['next_state'].append(observation)
    data['reward'].append(reward)

    #only used for the Cave environment
    if reward > 24:
       observation, info = env.reset()
       


env.close()
print(len(data['state']))
print(len(data['action']))


print("Saving data")
with h5py.File('expert_data/CaveUpdated.h5', 'w') as f:
    f.create_dataset('states', data= data['state'], compression='gzip', compression_opts=9)
    f.create_dataset('actions', data=data['action'])
    f.create_dataset('next_states', data=data['next_state'], compression='gzip', compression_opts=9)
    f.create_dataset('reward', data=data['reward'])
   
    






