import json
import numpy as np
from PIL import Image
import pickle

with open("data/Tree.bin", "rb") as binary_file:
    data = pickle.load(binary_file)

action = data['action']
print(len(action))

# array = np.array(state[90])

# array = array.astype(np.uint8)

# image = Image.fromarray(array)
# image.show()
