from imports import *
from model import AutoEncoder
from dataset import mnist_dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

loader = mnist_dataset(batch_size=32, train=False)
dataiter = iter(loader)
images, _ = next(dataiter)

images = images.view(-1, 28 * 28).to(device)
with torch.no_grad():
    reconstructed = model(images)

fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
for i in range(10):
    axes[0, i].imshow(images[i].cpu().reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i].cpu().reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
plt.show()
