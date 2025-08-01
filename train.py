from einops import rearrange
import torch

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