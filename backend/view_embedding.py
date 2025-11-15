import torch
import sys

path = sys.argv[1]
tensor = torch.load(path)
print("Shape:", tensor.shape)
print(tensor)
