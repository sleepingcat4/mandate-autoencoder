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

def train_conv_autoencoder(model, loader, loss_fn, optimizer, epochs, device):
    """
    Specialized training function for Conv_Autoencoder model.
    Handles the flattened input format required by Conv_Autoencoder.
    
    Args:
        model: Conv_Autoencoder instance
        loader: DataLoader providing batches of (images, labels)
        loss_fn: Loss function (e.g., nn.MSELoss())
        optimizer: Optimizer instance
        epochs: Number of training epochs
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        List of loss values recorded during training
    """
    model.to(device)
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in loader:
            # Conv_Autoencoder expects flattened input (batch_size, input_dim)
            images = images.view(images.size(0), -1).to(device)
            
            # Forward pass
            output = model(images)
            loss = loss_fn(output, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss
            losses.append(loss.item())
            epoch_loss += loss.item()

        print(f"ConvAE Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    # Save trained model
    torch.save(model.state_dict(), f"conv_autoencoder.pth")
    print(f"ConvAutoencoder saved as conv_autoencoder.pth")
    return losses

def train_vae(model, loader, optimizer, epochs, device):
    """
    Training function for Variational Autoencoder (VAE) model.
    Tracks and reports both total loss and its components (BCE + KLD).
    
    Args:
        model: VAE model instance
        loader: DataLoader providing batches of (images, labels)
        optimizer: Optimizer instance (e.g., Adam)
        epochs: Number of training epochs
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Trained model instance
    """
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0# Track total loss
        epoch_bce = 0 # Track reconstruction loss component
        epoch_kld = 0# Track KL divergence component
        
        for images, _ in loader:# Ignore labels for unsupervised learning
            images = images.to(device)
            recon, mu, log_var = model(images, epoch=epoch)# Forward pass - returns reconstruction, mean and log variance
            
            loss = model.loss_function(recon, images, mu, log_var) # Compute loss using VAE's custom loss function
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate and record loss components for monitoring
            with torch.no_grad():# No need for gradients during evaluation
                bce = F.binary_cross_entropy(recon, images, reduction='sum') # Binary Cross Entropy (reconstruction loss)
                kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())# KL Divergence (latent space regularization)
                epoch_loss += loss.item() * images.size(0)# Accumulate losses (multiply by batch size for proper averaging)
                epoch_bce += bce.item()
                epoch_kld += kld.item()
        
        # Calculate epoch averages and print progress
        avg_loss = epoch_loss / len(loader.dataset)
        print(f'Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | '
              f'BCE: {epoch_bce/len(loader.dataset):.4f} | '
              f'KLD: {epoch_kld/len(loader.dataset):.4f}')
    
    # Save final model after training completes
    torch.save(model.state_dict(), "vae.pth")
    print("Model saved as vae.pth")
    return model