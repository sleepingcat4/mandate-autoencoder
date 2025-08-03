from model import AutoEncoder, Sparse_Autoencoder,Conv_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = Conv_Autoencoder
model_path = "Conv_Autoencoder.pth"
model_kwargs = dict(input_dim=784,latent_dim=9)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device)
