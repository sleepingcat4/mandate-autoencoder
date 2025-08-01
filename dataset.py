from imports import * 

def mnist_dataset(batch_size=32, root="./data", train=True, shuffle=True):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader 