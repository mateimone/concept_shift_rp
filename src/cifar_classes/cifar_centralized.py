from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Subset, Dataset


class ListDataset(Dataset):
    """Dataset wrapping list of tuples."""
    def __init__(self, data):
        """
        Args:
            data (list of tuples): A list where each tuple is of the form (image, label).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = F.interpolate(batch_X, size=(224, 224), mode='bilinear', align_corners=False).to(device)

            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total


def get_client_loader(batch_size=64):
    # client_dataset = Subset(train_dataset, chosen_indices)

    # subset_list = [(extra_transform(client_dataset[i][0]), client_dataset[i][1]) for i in range(len(client_dataset))]
    subset_list = []
    j = 0
    for i in range(len(train_dataset)):
        if i not in chosen_indices:
            subset_list.append((train_transform(train_dataset[i][0]), train_dataset[i][1]))
        else:
            j += 1
            subset_list.append((train_transform_mix(train_dataset[i][0]), train_dataset[i][1]))

    dataset = ListDataset(subset_list)
    print(j)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

global_model = models.resnet50(pretrained=True).to(device)
for param in global_model.parameters():
    param.requires_grad = False

num_ftrs = global_model.fc.in_features
global_model.fc = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_ftrs, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
).to(device)

train_transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_transform_mix = v2.Compose([
    v2.ColorJitter(brightness=0.5, contrast=0.4),
    v2.RandomGrayscale(p=0.5),
    v2.ToTensor(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

chosen_indices = []
with open("../../indices_to_drift.csv", newline='') as lil_file:
    reader = csv.reader(lil_file)
    for row in reader:
        chosen_indices = [int(item) for item in row]
        break

train_dataset = CIFAR10("../../data/cifar", train=True, download=True, transform=None)
test_dataset = CIFAR10("../../data/cifar", train=False, download=True, transform=train_transform_mix)


train_loader = get_client_loader()
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

num_epochs = 40

train_accuracies = []
test_accuracies = []

global_model.train()

optimizer = optim.Adam(global_model.parameters(), lr=0.00002)

for epoch in tqdm(range(num_epochs)):
    global_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = F.interpolate(batch_X, size=(224, 224), mode='bilinear', align_corners=False).to(device)

        optimizer.zero_grad()
        outputs = global_model(batch_X)
        loss = nn.CrossEntropyLoss()(outputs, batch_y)

        loss.backward()
        optimizer.step()

    train_accuracy = evaluate_model(global_model, train_loader)
    test_accuracy = evaluate_model(global_model, test_loader)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs} - Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')

# final_train_accuracy = evaluate_model(global_model, train_loader)
# final_test_accuracy = evaluate_model(global_model, test_loader)

# print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
# print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')

torch.save(global_model, "../FrozenModels/cifar_color_jitter/cifar_centralized")

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Centralized Learning: Training and Test Accuracy Over Epochs - Normal Training Datan ')
plt.legend()
plt.grid(True)
plt.show()