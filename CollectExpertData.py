import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import json
import pickle



data = dict()
data['state'] = []
data['next_state'] = []
data['reward'] = []
data['action'] = []
env_name = input("What environment should the agent play in? ")
action_space = dict()
filename = env_name + ".bin"


if(env_name == "Tree"):
    env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 512, obs_height = 512)
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

counter = 0
for t in range(100):
    
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

    if(counter > 5):
        break
    
    #only used for the Cave environment
    if reward > 24:
        if(counter == 5):
            break
        counter += 1
        observation, info = env.reset()
        continue

env.close()

with open(filename, "wb") as binary_file:
    pickle.dump(data, binary_file)
   
    






