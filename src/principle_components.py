import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy
import matplotlib.pyplot as plt

transform = v2.ToTensor()

transform_other_mix = v2.Compose([
    v2.ColorJitter(brightness=1, contrast=0.9),
    v2.RandomGrayscale(p=1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
])


cifar10_train = datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=transform)
cifar10_gray_train = datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=transform_other_mix)


batch_size = 50000  # all data in a single batch for PCA
cifar10_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=False)
cifar10_gray_loader = DataLoader(cifar10_gray_train, batch_size=batch_size, shuffle=False)


cifar10_images, _ = next(iter(cifar10_loader))
cifar10_gray_images, _ = next(iter(cifar10_gray_loader))


cifar10_images_flattened = cifar10_images.view(cifar10_images.size(0), -1).numpy()
cifar10_gray_flattened = cifar10_gray_images.view(cifar10_gray_images.size(0), -1).numpy()


n_components = 3072
pca_original = PCA(n_components=n_components)
pca_gray = PCA(n_components=n_components)

pca_original.fit(cifar10_images_flattened)
pca_gray.fit(cifar10_gray_flattened)


def print_cumulative_variance(explained_variance, dataset_name):
    cumulative_variance = 0
    print(f"Cumulative Explained Variance for {dataset_name} Dataset:")
    for i, variance in enumerate(explained_variance):
        cumulative_variance += variance
        print(f"Component {i+1}: {cumulative_variance:.6f}")

print_cumulative_variance(pca_original.explained_variance_ratio_, "CIFAR-10")
print_cumulative_variance(pca_gray.explained_variance_ratio_, "Transformed CIFAR-10")
