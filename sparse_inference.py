from model import Sparse_Autoencoder
from imports import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Sparse_Autoencoder(input_dim=784, hidden_dim=128, sparsity_level=0.05, lambda_sparse=1e-3)
model.load_state_dict(torch.load("sparse_autoencoder.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.ToTensor()
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

images, _ = next(iter(test_loader))
images = images.view(images.size(0), -1).to(device)

with torch.no_grad():
    recon = model(images)

images = images.view(-1, 1, 28, 28).cpu()
recon = recon.view(-1, 1, 28, 28).cpu()

fig, axs = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axs[0, i].imshow(images[i].squeeze(), cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].imshow(recon[i].squeeze(), cmap='gray')
    axs[1, i].axis('off')
axs[0, 0].set_ylabel("Input", fontsize=12)
axs[1, 0].set_ylabel("Reconst.", fontsize=12)
plt.tight_layout()
plt.show()
