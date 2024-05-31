# import os
# import copy
# import time
# import pickle
# import numpy as np
# from tqdm import tqdm
#
# import torch
#
# import src.models
# from src.utils import fed_avg
# from torch.utils.data import DataLoader, Subset
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from src.utils import split_dirichlet
# import torch.optim as optim
# import matplotlib.pyplot as plt
#
#
# def evaluate_model(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_X, batch_y in loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             outputs = model(batch_X)
#             _, predicted = torch.max(outputs, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
#     return correct / total
#
#
#
#
# def get_client_loader(client_id, batch_size=100):
#     client_dataset = Subset(train_dataset, client_indices[client_id])
#     return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
#
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
# global_model = models.resnet50(pretrained=True).to(device)
# for param in global_model.parameters():
#     param.requires_grad = False
#
# num_ftrs = global_model.fc.in_features
# global_model.fc = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     torch.nn.Linear(num_ftrs, 1024),
#     torch.nn.ReLU(),
#     torch.nn.Linear(1024, 512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 10),
# ).to(device)
# # global_model = src.models.CNNCifar().to(device)
#
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
#
# train_dataset = CIFAR10("../data/cifar", train=True, download=True, transform=transform)
# test_dataset = CIFAR10("../data/cifar", train=False, download=True, transform=transform)
#
# num_clients = 10
# num_rounds = 20
# clients_per_round = 2
# num_epochs = 5
#
# # train_dataset.train_labels = torch.tensor([train_dataset[i][1].item() for i in range(len(train_dataset))], dtype=torch.float32)
# client_indices = split_dirichlet(train_dataset, num_clients, is_cfar=True, beta=0.5)
# client_indices = {k: np.array(v, dtype=int) for k, v in client_indices.items()}
#
# train_accuracies = []
# val_accuracies = []
# test_accuracies = []
#
# global_model.train()
#
# for round_num in tqdm(range(num_rounds)):
#     selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)
#
#     dataset_size_per_client = [len(client_indices[i]) for i in selected_clients]
#     local_models = []
#     for client_id in selected_clients:
#         client_loader = get_client_loader(client_id)
#         local_model = copy.deepcopy(global_model).to(device)
#         optimizer = optim.Adam(local_model.parameters(), lr=0.001)
#         local_model.train()
#
#         for epoch in range(num_epochs):
#             for batch_X, batch_y in client_loader:
#                 batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                 optimizer.zero_grad()
#                 outputs = local_model(batch_X)
#                 loss = nn.CrossEntropyLoss()(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#
#         local_models.append(local_model.state_dict())
#
#     global_weights = fed_avg(local_models, dataset_size_per_client)
#     global_model.load_state_dict(global_weights)
#
#     train_accuracy = evaluate_model(global_model, DataLoader(train_dataset, batch_size=100, shuffle=False))
#     # val_accuracy = evaluate_model(global_model, val_loader)
#     test_accuracy = evaluate_model(global_model, DataLoader(test_dataset, batch_size=100, shuffle=True))
#
#     train_accuracies.append(train_accuracy)
#     # val_accuracies.append(val_accuracy)
#     test_accuracies.append(test_accuracy)
#
#     print(f'Round {round_num + 1}/{num_rounds} - Train Accuracy: {train_accuracy * 100:.2f}%, '
#           # f'Val Accuracy: {val_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')
#           f'No val accuracy, Test Accuracy: {test_accuracy * 100:.2f}%')
#
# final_train_accuracy = evaluate_model(global_model, DataLoader(train_dataset, batch_size=100, shuffle=False))
# final_test_accuracy = evaluate_model(global_model, DataLoader(test_dataset, batch_size=100, shuffle=True))
#
# print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
# print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')
#
# rounds = range(1, num_rounds + 1)
# plt.figure(figsize=(10, 5))
# plt.plot(rounds, train_accuracies, label='Train Accuracy')
# # plt.plot(rounds, val_accuracies, label='Validation Accuracy')
# plt.plot(rounds, test_accuracies, label='Test Accuracy')
# plt.xlabel('Rounds')
# plt.ylabel('Accuracy')
# plt.title('Federated Learning: Training, Validation, and Test Accuracy Over Rounds')
# plt.legend()
# plt.grid(True)
# plt.show()


import os
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

import src.models  # Assuming this contains the CNNCifar model

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = CIFAR10("../data/cifar", train=True, download=True, transform=transform)
test_dataset = CIFAR10("../data/cifar", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

num_epochs = 5

train_accuracies = []
test_accuracies = []

global_model.train()

optimizer = optim.Adam(global_model.parameters(), lr=0.00002)

for epoch in tqdm(range(num_epochs)):
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

final_train_accuracy = evaluate_model(global_model, train_loader)
final_test_accuracy = evaluate_model(global_model, test_loader)

print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Centralized Learning: Training and Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
