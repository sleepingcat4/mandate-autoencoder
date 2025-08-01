import torch
import matplotlib.pyplot as plt
from einops import rearrange

def load_model(model_class, model_path, device, **model_kwargs):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def infer_and_visualize(model, loader, device, num_images=10):
    dataiter = iter(loader)
    images, _ = next(dataiter)
    images = images.view(images.size(0), -1).to(device)

    with torch.no_grad():
        reconstructed = model(images)

    images = images.view(-1, 1, 28, 28).cpu()
    reconstructed = reconstructed.view(-1, 1, 28, 28).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images*1.5, 3))
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.show()
