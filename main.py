import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([1, 2, 3])
tensor = tensor.to(device)

print(tensor, tensor.device)