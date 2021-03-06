import torch
from torchvision import datasets, transforms
def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    train = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    
    # Download and load the test data
    test = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    
    return train, test

def mnist_loader(train, test):
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return trainloader, testloader

