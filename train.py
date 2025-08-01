from imports import *

loader = mnist_dataset(batch_size=32)
model = AutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-13, betas=(0.8, 0.8999), weight_decay=1e-8)

epochs = 20 
outputs = []
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import trange 
pbar = trange(epochs, desc="Training Autoencoder")
for epoch in pbar:
    for images, _ in loader:
        reconstructed = model(images)
        loss = loss_function(reconstructed, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    outputs.append((epoch, images, reconstructed))
    pbar.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")