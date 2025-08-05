from einops import rearrange
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

def train_model(model, loader, loss_fn, optimizer, epochs, device):
    model.to(device)
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in loader:
            images = rearrange(images, 'b c h w -> b (c h w)').to(device)

            output = model(images)
            loss = loss_fn(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), f"{model.__class__.__name__}.pth")
    print(f"Model saved as {model.__class__.__name__}.pth")

    return losses

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def vqemaautoencoder_train(model, training_loader, data_variance, num_training_updates=15000, 
                           learning_rate=1e-3, device="cuda"):

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(num_training_updates):
        data, _ = next(iter(training_loader))
        data = data.to(device)

        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)

        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print(f"{i+1} iterations")
            print(f"recon_error: {np.mean(train_res_recon_error[-100:]):.3f}")
            print(f"perplexity: {np.mean(train_res_perplexity[-100:]):.3f}\n")

    torch.save(model.state_dict(), "vqema_autoencoder.pth")
    return train_res_recon_error, train_res_perplexity
