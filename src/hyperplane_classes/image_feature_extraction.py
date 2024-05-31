import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision.models as models
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shifted_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
])

original_transform = transforms.Compose([
    transforms.ToTensor(),
])

original_dataset = CIFAR10(root='../../data/cifar', train=True, download=False, transform=original_transform)
shifted_dataset = CIFAR10(root='../../data/cifar', train=True, download=False, transform=shifted_transform)

original_loader = DataLoader(original_dataset, batch_size=64, shuffle=True)
shifted_loader = DataLoader(shifted_dataset, batch_size=64, shuffle=True)

global_model = models.resnet50(pretrained=True).to(device)
for param in global_model.parameters():
    param.requires_grad = False

num_ftrs = global_model.fc.in_features
global_model.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device)

feature_extractor = nn.Sequential(*list(global_model.children())[:-1]).to(device)

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            output = feature_extractor(images)
            output = output.view(output.size(0), -1)  # Flatten features
            features.append(output.cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

original_features, original_labels = extract_features(original_loader)
shifted_features, shifted_labels = extract_features(shifted_loader)

pca = PCA(n_components=2)
original_pca_features = pca.fit_transform(original_features)
shifted_pca_features = pca.transform(shifted_features)

plt.figure(figsize=(8, 6))
plt.scatter(original_pca_features[:, 0], original_pca_features[:, 1], label='Original', alpha=0.5)
plt.scatter(shifted_pca_features[:, 0], shifted_pca_features[:, 1], label='Shifted', alpha=0.5)
plt.title('PCA of Original and Shifted Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

tsne = TSNE(n_components=2)
original_tsne_features = tsne.fit_transform(original_features)
shifted_tsne_features = tsne.fit_transform(shifted_features)

plt.figure(figsize=(8, 6))
plt.scatter(original_tsne_features[:, 0], original_tsne_features[:, 1], label='Original', alpha=0.5)
plt.scatter(shifted_tsne_features[:, 0], shifted_tsne_features[:, 1], label='Shifted', alpha=0.5)
plt.title('t-SNE of Original and Shifted Images')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
