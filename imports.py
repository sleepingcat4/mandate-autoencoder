from einops import rearrange
from torchvision import datasets, transforms 
import torch 
from torch import optim, nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader