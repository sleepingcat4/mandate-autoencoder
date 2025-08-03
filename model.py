from imports import *

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=9):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Sparse_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_level=0.05, lambda_sparse=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
        self.sparsity_level = sparsity_level
        self.lambda_sparse = lambda_sparse

    def forward(self, x):
        self.encoded = self.activation(self.encoder(x))
        x = self.output_activation(self.decoder(self.encoded))
        return x

    def sparse_loss(self, y_true, y_pred):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        mean_activation = torch.mean(self.encoded, dim=0)
        kl = self.sparsity_level * torch.log(self.sparsity_level / (mean_activation + 1e-10)) + \
             (1 - self.sparsity_level) * torch.log((1 - self.sparsity_level) / (1 - mean_activation + 1e-10))
        kl_div = torch.sum(kl)
        return mse_loss + self.lambda_sparse * kl_div

class Conv_Autoencoder(nn.Module):
    def __init__(self,input_dim,latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.img_size = int(input_dim**0.5)
        self.encoder = nn.Sequential(
            nn.Unflatten(1,(1,self.img_size,self.img_size)),
            nn.Conv2d(1, 16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1568,256),
            nn.ReLU(),
            nn.Linear(256,latent_dim)
        )
        self.decoder = nn.Sequential(
           nn.Linear(latent_dim,256),
           nn.ReLU(),
           nn.Linear(256,1568),
           nn.ReLU(),
           nn.Unflatten(1,(32,7,7)),
           nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,padding=1,output_padding=1),
           nn.Sigmoid(),
           nn.Flatten()
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded