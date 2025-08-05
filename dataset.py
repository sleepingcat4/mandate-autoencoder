from imports import *

def mnist_dataset(batch_size=32, root="./data", train=True, shuffle=True):
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=tensor_transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def cifar10_dataset(batch_size=32, root="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    ])
    
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, pin_memory=True)
    
    return train_loader, val_loader