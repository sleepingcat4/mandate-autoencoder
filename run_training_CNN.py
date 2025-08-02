from model import AutoEncoder, Sparse_Autoencoder,Conv_Autoencoder
from dataset import mnist_dataset
from torch import nn, optim
import torch
from train import train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)

model = Conv_Autoencoder(input_channels=1,latent_dim=64).to(device)


if isinstance(model, Sparse_Autoencoder):
    loss_fn = model.sparse_loss
else:
    loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
epochs = 20

train_model(model, loader, loss_fn, optimizer, epochs, device,flatten=False)

from pathlib import Path
from model import AutoEncoder, Sparse_Autoencoder,Conv_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = Conv_Autoencoder
model_path = "Conv_Autoencoder.pth"
model_kwargs = dict(input_channels=1, latent_dim=64)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device,num_images=5)
