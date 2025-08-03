from einops import rearrange
import torch
from typing import Callable, Tuple

def preprocess_data(images: torch.Tensor) -> torch.Tensor:
    """Flatten images from (B,C,H,W) to (B, C*H*W)."""
    return rearrange(images, 'b c h w -> b (c h w)')

def compute_loss(
    model: torch.nn.Module,
    loss_fn: Callable,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute loss based on model type (VAE or standard).
    Returns:
        - loss (torch.Tensor)
        - extra_info (dict, optional metadata like mu/log_var for VAE)
    """
    if hasattr(model, 'reparameterize'):
        output, mu, log_var = model(inputs)
        loss = loss_fn(output, inputs, mu, log_var)
        extra_info = {"mu": mu, "log_var": log_var}
    else:
        output = model(inputs)
        loss = loss_fn(output, inputs)
        extra_info = {}
    return loss, extra_info

def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    epoch_loss = 0.0
    for images, _ in loader:
        inputs = preprocess_data(images).to(device)
        
        optimizer.zero_grad()
        loss, _ = compute_loss(model, loss_fn, inputs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

def train_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> list:
    """Main training loop."""
    model.to(device)
    model.train()
    losses = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, loader, loss_fn, optimizer, device)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), f"{model.__class__.__name__}.pth")
    print(f"Model saved as {model.__class__.__name__}.pth")
    return losses