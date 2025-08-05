from model import VQE_AutoEncoder
from dataset import cifar10_dataset
from train import vqemaautoencoder_train
import torch
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 256
    num_training_updates = 15000
    learning_rate = 1e-3

    train_loader, _ = cifar10_dataset(batch_size=batch_size)
    train_set = train_loader.dataset
    data_variance = np.var(train_set.data / 255.0)

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99

    model = VQE_AutoEncoder(num_hiddens, num_residual_layers, num_residual_hiddens,
                            num_embeddings, embedding_dim, commitment_cost, decay)

    vqemaautoencoder_train(model, train_loader, data_variance, num_training_updates, learning_rate, device)

if __name__ == "__main__":
    main()
