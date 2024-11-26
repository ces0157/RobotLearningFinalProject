import json
import numpy as np
from PIL import Image
import pickle

with open("Tree.bin", "rb") as binary_file:
    data = pickle.load(binary_file)

state = data['state']

array = np.array(state[90])

array = array.astype(np.uint8)

image = Image.fromarray(array)
image.show()
