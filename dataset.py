from imports import *

def mnist_dataset(batch_size=32, root="./data", train=True, shuffle=True):
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=tensor_transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
