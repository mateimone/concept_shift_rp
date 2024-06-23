import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torchvision.transforms import v2
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def main():
    def extract_features(dataloader, model):
        with torch.no_grad():
            features = []
            for images, _ in dataloader:
                outputs = model(images)
                features.append(outputs.view(outputs.size(0), -1))
            features = torch.cat(features, dim=0)
        return features.numpy()

    def compute_histograms(features, bins=256):
        histograms = []
        for i in range(features.shape[1]):
            hist, _ = np.histogram(features[:, i], bins=bins, density=True)
            histograms.append(hist)
        return np.array(histograms)

    transform = v2.Compose([
        v2.ColorJitter(brightness=1, contrast=0.9),
        v2.RandomGrayscale(p=1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianBlur(3, (0.1, 2)),
        v2.ToTensor(),
        v2.GaussianNoise()
    ])

    trainset_original = datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=v2.Compose([v2.ToTensor()]))
    trainset_transformed = datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=transform)

    trainloader_original = torch.utils.data.DataLoader(trainset_original, batch_size=10000, shuffle=False, num_workers=2)
    trainloader_transformed = torch.utils.data.DataLoader(trainset_transformed, batch_size=10000, shuffle=False, num_workers=2)

    model = models.resnet50(pretrained=True)
    model = model.eval()
    model = nn.Sequential(*list(model.children())[:-1])

    features_original = extract_features(trainloader_original, model)
    features_transformed = extract_features(trainloader_transformed, model)

    hist_original = compute_histograms(features_original)
    hist_transformed = compute_histograms(features_transformed)

    hist_original += 1e-10
    hist_transformed += 1e-10

    js_divergences = []
    for i in range(hist_original.shape[0]):
        js_div = jensenshannon(hist_original[i], hist_transformed[i])
        js_divergences.append(js_div)

    average_js_divergence = np.mean(js_divergences)

    print(average_js_divergence)

if __name__ == "__main__":
    main()