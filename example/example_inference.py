from model import AutoEncoder, Sparse_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = Sparse_Autoencoder
model_path = "sparse_autoencoder.pth"
model_kwargs = dict(input_dim=784, hidden_dim=128, sparsity_level=0.05, lambda_sparse=1e-3)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device)
