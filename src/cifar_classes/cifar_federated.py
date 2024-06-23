import copy
import random
import csv

import numpy as np
from tqdm import tqdm

import torch

from src.utils import fed_avg
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from src.sampling import split_dirichlet, cifar_iid
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2


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


def get_client_loader(client_id: int, train_dataset, client_indices, train_transform_mix, train_transform, drifted=False, batch_size=64):
    client_dataset = Subset(train_dataset, client_indices[client_id])

    transform = train_transform_mix if drifted else train_transform
    subset_list = [(transform(client_dataset[i][0]), client_dataset[i][1]) for i in range(len(client_dataset))]
    dataset = ListDataset(subset_list)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(model_num):
    global device
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

    train_dataset = CIFAR10("../../data/cifar", train=True, download=True, transform=None)
    test_dataset = CIFAR10("../../data/cifar", train=False, download=True, transform=train_transform_mix)

    num_clients = 10
    num_rounds = 20
    clients_per_round = 2
    num_epochs = 5

    # client_indices = split_dirichlet(train_dataset, num_clients, is_cfar=True, beta=0.5)
    client_indices = cifar_iid(train_dataset, num_clients)
    client_indices = {k: np.array(v, dtype=int) for k, v in client_indices.items()}

    train_accuracies = []
    test_accuracies = []

    global_model.train()

    chosen_clients = random.sample(range(num_clients), 2)

    # with open("../../indices_to_drift.csv", mode='w') as lil_file:
    #     writer = csv.writer(lil_file)
    #     row = []
    #     for chosen_client in chosen_clients:
    #         row = row + [client_indices[chosen_client][i] for i in range(len(client_indices[chosen_client]))]
    #
    #         print(len(client_indices[chosen_client]))
    #
    #     print(len(row))
    #     writer.writerow(row)

    for round_num in tqdm(range(num_rounds)):
        selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)

        dataset_size_per_client = [len(client_indices[i]) for i in selected_clients]
        local_models = []
        print(f"Selected clients {selected_clients}")
        for client_id in selected_clients:
            drifted = False
            if client_id in chosen_clients:
                drifted = True
            client_loader = get_client_loader(client_id, train_dataset, client_indices, train_transform_mix, train_transform, drifted)
            # print(len(client_loader))
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.Adam(local_model.parameters(), lr=0.00002)
            local_model.train()

            for epoch in range(num_epochs):
                for batch_X, batch_y in client_loader:
                    # print(len(batch_X))
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_X = F.interpolate(batch_X, size=(224, 224), mode='bilinear', align_corners=False).to(device)
                    optimizer.zero_grad()

                    outputs = local_model(batch_X)
                    loss = nn.CrossEntropyLoss()(outputs, batch_y)

                    loss.backward()
                    optimizer.step()

            local_models.append(local_model.state_dict())

        global_weights = fed_avg(local_models, dataset_size_per_client)
        global_model.load_state_dict(global_weights)

        train_accuracy = evaluate_model(global_model, DataLoader(CIFAR10("../../data/cifar", train=False, download=True, transform=train_transform), batch_size=100, shuffle=False))
        test_accuracy = evaluate_model(global_model, DataLoader(test_dataset, batch_size=100, shuffle=True))

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Round {round_num + 1}/{num_rounds} - Train Accuracy: {train_accuracy * 100:.2f}%, '
              f'No val accuracy, Test Accuracy: {test_accuracy * 100:.2f}%')

    torch.save(global_model, f"../FrozenModels/cifar_color_jitter/iid_cifar")

    # final_train_accuracy = evaluate_model(global_model, DataLoader(train_dataset, batch_size=100, shuffle=False))
    final_test_accuracy = evaluate_model(global_model, DataLoader(test_dataset, batch_size=100, shuffle=True))

    # print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
    print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')

    rounds = range(1, num_rounds + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, train_accuracies, label='Train Accuracy')
    plt.plot(rounds, test_accuracies, label='Test Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')

    plt.title(
        'non-IID Federated Learning: 20% "Train" Transformed Data, Tested on "Train" Transforms, 2 Clients/Round')
    plt.legend()
    plt.grid(True)
    plt.show()

main(1)
