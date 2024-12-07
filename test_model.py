import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import torch
from torchvision import transforms
from PIL import Image
from dagger import Learner
import os




def main():
    env_name = input("What environment should we Test the Expert on  (Tree, Cave, Spider)? ")
    if(env_name == "Cave"):
        model = Learner(7)
        if os.path.exists('Great_model2.pth'):
            model.load_state_dict(torch.load('Great_model2.pth', weights_only=True))
            model.eval()
            env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512, frameskip=2)
            env.mouse_mov = .25
    
        else:
            print("No model found")
            return
    elif(env_name == "Tree"):
        model = ExpertModel(8)
        if os.path.exists('Expert_model_Tree.pth'):
            model.load_state_dict(torch.load('Expert_model_Tree.pth', weights_only=True))
            model.eval()
            env = gym.make("Craftium/ChopTree-v0", render_mode = "human", obs_width = 512, obs_height = 512)

    elif(env_name == "Spider"):
        model = ExpertModel(10)
        if os.path.exists('Expert_model_Spider.pth'):
            model.load_state_dict(torch.load('Expert_model_Spider.pth', weights_only=True))
            model.eval()
            env = gym.make("Craftium/SpidersAttack-v0", render_mode = "human", obs_width = 512, obs_height = 512)
            
    else:
        print("Invalid Environment")
        return
    
    
    #we need to preprocess the image before giving it to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels for better efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    observation, info = env.reset()

    for i in range(2000):  
        observation = np.array(observation)
        observation = Image.fromarray(observation)
        
        observation = transform(observation)
        
        _, action = torch.max(model(observation), 1)
        action = action[0].item()

        #print(action_space[action])
        #user_input = input("Press Enter to continue")

        observation, reward, terminated, truncated, _info = env.step(action)
        
    env.close()



   


if __name__ == "__main__":
    main()
