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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    
class VQE_AutoEncoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super().__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=8.0):  # Set beta to 8. A smaller beta value will make KLD particularly small
        """
        Variational Autoencoder (VAE) implementation with enhanced architecture.
        Uses KL divergence weighting (β-VAE) for better latent space organization.
        
        Args:
            input_dim (int): Dimension of input data
            latent_dim (int): Dimension of latent space representation
            beta (float): Weight for KL divergence term (default: 8.0)
                          Higher values encourage more disentangled latent representations
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta # β parameter for β-VAE formulation

        # Enhanced encoder network with layer normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),# LeakyReLU helps prevent dead neurons
            nn.Linear(512, 256),
            nn.LayerNorm(256),# LayerNorm stabilizes training
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2)
        )
        
        # Latent spatial parameter layer
        self.fc_mu = nn.Linear(128, latent_dim)# Mean of latent distribution
        self.fc_var = nn.Linear(128, latent_dim) # Log variance of latent distribution
        
        # Initialization
        nn.init.constant_(self.fc_var.bias, 0.0)  # Initialize log_var near 0 (σ≈1)
        nn.init.normal_(self.fc_var.weight, mean=0, std=0.001)# Small weights
        nn.init.constant_(self.fc_mu.bias, 0.0)# Zero bias for mean
        nn.init.normal_(self.fc_mu.weight, mean=0, std=0.001)# Small weights

        # Enhanced decoder network with layer normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid()# Output values between 0-1 for reconstruction
        )

    def encode(self, x):
        """
        Encodes input into parameters of latent distribution.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        h = self.encoder(x.view(-1, self.input_dim))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to enable backpropagation through sampling.
        Includes enhanced noise intensity for better exploration.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector with reparameterization
        """
        std = torch.exp(0.5 * log_var.clamp(min=-10, max=2))  # Constrained σ between (0.006,2.718)
        eps = torch.randn_like(std) * 1.2  # Enhanced noise multiplier (1.2x)
        return mu + eps * std

    def forward(self, x, epoch=None):
        """
        Forward pass of the VAE.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            epoch: Optional epoch number (unused in this implementation)
            
        Returns:
            During training: Tuple of (reconstruction, mu, log_var)
            Otherwise: Just reconstruction
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z).view_as(x) if not self.training else (self.decoder(z).view_as(x), mu, log_var)

    def loss_function(self, recon_x, x, mu, log_var):
        """
        Computes VAE loss = reconstruction loss + β*KL divergence.
        Includes additional constraint on latent mean for regularization.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Total loss averaged over batch
        """
        # Binary cross-entropy reconstruction loss
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # Enhanced KL divergence with additional mean constraint
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        KLD += 0.5 * torch.mean(mu.pow(2))  # Additional constraint on mean
        # Weighted combination with β factor
        return (BCE + self.beta * KLD) / x.size(0)# Average over batch
class Conv_Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Convolutional Autoencoder implementation for image data.
        
        Args:
            input_dim (int): Total dimension of flattened input (img_size * img_size)
            latent_dim (int): Dimension of the bottleneck latent representation
        """
        super().__init__()
        self.input_dim = input_dim
        self.img_size = int(input_dim**0.5)  # Calculate original image size (assuming square)
        
        # Encoder network: Compresses input image into latent representation
        self.encoder = nn.Sequential(
            # Reshape flattened input into image format (batch, channel, height, width)
            nn.Unflatten(1, (1, self.img_size, self.img_size)),  # [B,1,28,28] for MNIST
            
            # First convolutional block (halves spatial dimensions)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B,16,14,14]
            nn.ReLU(),
            
            # Second convolutional block (halves spatial dimensions again)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B,32,7,7]
            nn.ReLU(),
            
            # Flatten before dense layers
            nn.Flatten(),
            
            # First fully-connected layer
            nn.Linear(32*7*7, 256),  # 32*7*7=1568 for MNIST
            nn.ReLU(),
            
            # Final projection to latent space
            nn.Linear(256, latent_dim)
        )
        
        # Decoder network: Reconstructs image from latent representation
        self.decoder = nn.Sequential(
            # Expand latent vector
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            
            # Project to conv transpose input size
            nn.Linear(256, 32*7*7),  # Matching encoder's last conv output shape
            nn.ReLU(),
            
            # Reshape for convolutional transpose operations
            nn.Unflatten(1, (32, 7, 7)),  # [B,32,7,7]
            
            # First transposed convolution block (doubles spatial dimensions)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # [B,16,14,14]
            nn.ReLU(),
            
            # Second transposed convolution block (doubles spatial dimensions)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # [B,1,28,28]
            nn.Sigmoid(),  # Pixel values between 0-1
            
            # Flatten output to match input format
            nn.Flatten()
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Flattened input tensor of shape [batch_size, input_dim]
            
        Returns:
            Reconstructed output tensor of same shape as input
        """
        encoded = self.encoder(x)  # Encode to latent space
        decoded = self.decoder(encoded)  # Reconstruct from latent space
        return decoded