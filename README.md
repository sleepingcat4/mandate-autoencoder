#### Mandate Autoencoder
![autoencoder image](auto.png)
*results from our trained autoencoder*

Introducing a high-level python wrapper to train and perform inference a linear autoencoder. There are different types of autoencoder and most popular tutorials focus on ConvNet Autoencoder including UToronto (https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html)

Linear or original Dense network based autoencoder is the primary autoencoder which uses the fundamental principle of Neural Networks and compress an image and decompress it to keep only the high-level features. We have made this repo to have it customizable and train a network with minimum effort. 

##### How to use it?
Download this repo and go inside it. Then 
Basic Autoencoder Usage
```Python 
from model import AutoEncoder, Sparse_Autoencoder
from dataset import mnist_dataset
from torch import nn, optim
import torch
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)

model = Sparse_Autoencoder(input_dim=784, hidden_dim=128, sparsity_level=0.05, lambda_sparse=1e-3)

if isinstance(model, Sparse_Autoencoder):
    loss_fn = model.sparse_loss
else:
    loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
epochs = 20

train_model(model, loader, loss_fn, optimizer, epochs, device)

```

Perform inference 
```Python
from model import AutoEncoder, Sparse_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = Sparse_Autoencoder
model_path = "sparse_autoencoder.pth"
model_kwargs = dict(input_dim=784, hidden_dim=128, sparsity_level=0.05, lambda_sparse=1e-3)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device)

````
Convolutional Autoencoder Usage
```Python
from model import Conv_Autoencoder
from dataset import mnist_dataset
from torch import nn, optim
import torch
from train import train_conv_autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)

model = Conv_Autoencoder(input_dim=784, latent_dim=32)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
epochs = 20

train_conv_autoencoder(model, loader, loss_fn, optimizer, epochs, device)
````
````python
from model import AutoEncoder, Sparse_Autoencoder,VAE,Conv_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = Conv_Autoencoder
model_path = "Conv_Autoencoder.pth"
model_kwargs = dict(input_dim=784, latent_dim=32)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device)
````
Variational Autoencoder (VAE) Usage
````python
from model import VAE
from dataset import mnist_dataset
import torch
from train import train_vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32)

model = VAE(input_dim=784, latent_dim=64, beta=8.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
epochs = 20

train_vae(model, loader, optimizer, epochs=epochs, device=device)
````
````python
from model import AutoEncoder, Sparse_Autoencoder,VAE,Conv_Autoencoder
from dataset import mnist_dataset
from inference import load_model, infer_and_visualize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = mnist_dataset(batch_size=32, train=False)

model_class = VAE
model_path = "vae.pth"
model_kwargs = dict(input_dim=784, latent_dim=64)

model = load_model(model_class, model_path, device, **model_kwargs)
infer_and_visualize(model, loader, device)
````
#### Trained model
We also provide our trained models in the model folder called **autoencoder.pth** and sparse_autoencoder.pth

## Upcoming Autoencoder Enhancements

- [x] Include Sparse Autoencoder
- [ ] Include Variational Autoencoder (VAE)
- [x] Include Convolutional Autoencoder
- [x] Refactor into high-level wrapper / framework
- [ ] Increase architectural difficulty and depth
- [ ] Add multi-GPU training support
- [ ] Extend to Denoising Autoencoder
- [ ] Add segmentation capabilities
- [ ] Support text input (e.g., embeddings)
- [ ] Support video input (e.g., 3D Conv or CNN+RNN)

