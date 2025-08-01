from imports import *

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=9):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)')
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = rearrange(x_hat, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return x_hat
