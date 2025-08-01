from model import Sparse_Autoencoder
from imports import *
from dataset import mnist_dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)
model = Sparse_Autoencoder(input_dim=784, hidden_dim=128, sparsity_level=0.05, lambda_sparse=1e-3)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for batch, _ in loader:
        batch = batch.view(batch.size(0), -1).to(device)
        output = model(batch)
        loss = model.sparse_loss(batch, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

torch.save(model.state_dict(), "sparse_autoencoder.pth")
print("Model saved to sparse_autoencoder.pth")