from model import AutoEncoder, Sparse_Autoencoder,Conv_Autoencoder
from dataset import mnist_dataset
from torch import nn, optim
import torch
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)

model = Conv_Autoencoder(input_dim=28*28,latent_dim=9)

if isinstance(model, Sparse_Autoencoder):
    loss_fn = model.sparse_loss
else:
    loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
epochs = 20

train_model(model, loader, loss_fn, optimizer, epochs, device)


