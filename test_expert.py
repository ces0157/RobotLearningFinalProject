import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
import torch
from torchvision import transforms
from PIL import Image
from train_expert import ExpertModel
import os



def main():
    env_name = input("What environment should we Test the Expert on  (Tree, Cave)? ")
    if(env_name == "Cave"):
        model = ExpertModel(7)
        if os.path.exists('Learner_model_Cave2.pth'):
            model.load_state_dict(torch.load('Learner_model_Cave2.pth', weights_only=True))
            model.eval()
            env = gym.make("Craftium/Speleo-v0", render_mode = "human", obs_width = 512, obs_height = 512)
    
        else:
            print("No model found")
            return
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
