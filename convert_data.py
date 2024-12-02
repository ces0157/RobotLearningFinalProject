import h5py
import numpy as np
import pickle

data1 = pickle.load(open("expert_data/Cave3.bin", "rb"))

states = data1['state']
actions = data1['action']
next_states = data1['next_state']
reward = data1['reward']

with h5py.File('expert_data/Cave3.h5', 'w') as f:
    f.create_dataset('states', data=states, compression='gzip', compression_opts=9)
    f.create_dataset('actions', data=actions)
    f.create_dataset('next_states', data=next_states, compression='gzip', compression_opts=9)
    f.create_dataset('reward', data=reward)


