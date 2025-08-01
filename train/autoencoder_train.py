from imports import *
from model import AutoEncoder
from dataset import mnist_dataset
from einops import rearrange
import os

loader = mnist_dataset(batch_size=32)
model = AutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

epochs = 20
outputs = []
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    for images, _ in loader:
        images = rearrange(images, 'b c h w -> b (c h w)').to(device)

        reconstructed = model(images)
        loss = loss_function(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    outputs.append((epoch, images, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved as autoencoder.pth")
