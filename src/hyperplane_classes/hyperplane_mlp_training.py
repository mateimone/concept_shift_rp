import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from src.sampling import  river_iid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RiverNDDataset(Dataset):
    def __init__(self, file_path, n_features):
        data = pd.read_csv(file_path)
        self.features = torch.tensor(data.iloc[:, 0:n_features].values, dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, n_features].values, dtype=torch.float32).unsqueeze(1)
        self.train_labels = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return features, label


full_train_dataset = RiverNDDataset(
    "../hyperplanes/30_samples_hyperplane/30_boundary_rotated_0_hyperplane.csv", 2)
# test_dataset = RiverNDDataset("../hyperplanes/30_boundary_rotated_10_hyperplane.csv", 2)
test_dataset = RiverNDDataset("../hyperplanes/100_samples_hyperplane/100_boundary_rotated_0_hyperplane.csv", 2)

train_size = len(full_train_dataset)
train_dataset = full_train_dataset
# train_size = int(0.9 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


class RiverNDMLP(nn.Module):
    def __init__(self, n_features):
        super(RiverNDMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.drop(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = torch.sigmoid(self.fc2(x))
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fed_avg(local_weights, dataset_size_per_client):
    avg_dict = {}
    sum_dataset = sum(dataset_size_per_client)
    for i, dictionary in enumerate(local_weights):
        for key, tensor in dictionary.items():
            if key not in avg_dict:
                avg_dict[key] = tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
            else:
                avg_dict[key] += tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
    return avg_dict


# Function to evaluate the model
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == batch_y).float().sum().item()
            total += batch_y.size(0)
    return correct / total


num_clients = 10
train_dataset.train_labels = torch.tensor([train_dataset[i][1].item() for i in range(len(train_dataset))], dtype=torch.float32)
# client_indices = split_dirichlet(train_dataset, num_clients, is_cfar=False, beta=0.5)
client_indices = river_iid(train_dataset, num_clients)
client_indices = {k: np.array(v, dtype=int) for k, v in client_indices.items()}


def get_client_loader(client_id, batch_size=100):
    client_dataset = Subset(train_dataset, client_indices[client_id])
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)


num_rounds = 20
clients_per_round = 2
num_epochs = 5

global_model = RiverNDMLP(2).to(device)

train_accuracies = []
val_accuracies = []
test_accuracies = []

for round_num in range(num_rounds):
    selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)
    dataset_size_per_client = [len(client_indices[i]) for i in selected_clients]
    local_models = []
    for client_id in selected_clients:
        client_loader = get_client_loader(client_id)
        local_model = RiverNDMLP(2).to(device)
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=0.0001)

        local_model.train()
        for epoch in range(num_epochs):
            for batch_X, batch_y in client_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = nn.BCELoss()(outputs, batch_y)
                loss.backward()
                optimizer.step()

        local_models.append(local_model.state_dict())

    # global_state_dict = global_model.state_dict()
    # for key in global_state_dict.keys():
    #     global_state_dict[key] = torch.stack([local_model[key].float() for local_model in local_models], 0).mean(0)
    global_weights = fed_avg(local_models, dataset_size_per_client)
    global_model.load_state_dict(global_weights)

    train_accuracy = evaluate_model(global_model, DataLoader(train_dataset, batch_size=100, shuffle=False))
    # val_accuracy = evaluate_model(global_model, val_loader)
    test_accuracy = evaluate_model(global_model, test_loader)

    train_accuracies.append(train_accuracy)
    # val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

    print(f'Round {round_num + 1}/{num_rounds} - Train Accuracy: {train_accuracy * 100:.2f}%, '
          # f'Val Accuracy: {val_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')
          f'No val accuracy, Test Accuracy: {test_accuracy * 100:.2f}%')

torch.save(global_model, '../FrozenModels/30_samples/30_iid_model_no_val_5')

final_train_accuracy = evaluate_model(global_model, DataLoader(train_dataset, batch_size=100, shuffle=False))
final_test_accuracy = evaluate_model(global_model, test_loader)

print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')

rounds = range(1, num_rounds + 1)
plt.figure(figsize=(10, 5))
plt.plot(rounds, train_accuracies, label='Train Accuracy')
# plt.plot(rounds, val_accuracies, label='Validation Accuracy')
plt.plot(rounds, test_accuracies, label='Test Accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Federated Learning: Training, Validation, and Test Accuracy Over Rounds')
plt.legend()
plt.grid(True)
plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, random_split
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# class RiverNDDataset(Dataset):
#     def __init__(self, file_path, n_features):
#         data = pd.read_csv(file_path)
#         self.features = torch.tensor(data.iloc[:, 0:n_features].values, dtype=torch.float32)
#         self.labels = torch.tensor(data.iloc[:, n_features].values, dtype=torch.float32).unsqueeze(1)
#         self.train_labels = self.labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         features = self.features[idx]
#         label = self.labels[idx]
#         return features, label
#
#
# full_train_dataset = RiverNDDataset("../hyperplanes_with_redundancy/30_samples_hyperplane_red_2/30_boundary_rotated_0_hyperplane.csv", 4)
# test_dataset = RiverNDDataset(
#     "../hyperplanes_with_redundancy/10k_samples_hyperplane/10k_boundary_rotated_0_hyperplane.csv", 4)
#
# train_size = len(full_train_dataset)
# train_dataset = full_train_dataset
# # train_size = int(0.8 * len(full_train_dataset))
# # val_size = len(full_train_dataset) - train_size
# # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
#
#
# class RiverNDMLP(nn.Module):
#     def __init__(self, n_features):
#         super(RiverNDMLP, self).__init__()
#         self.fc1 = nn.Linear(n_features, 1000)
#         self.bn1 = nn.BatchNorm1d(1000)
#         self.fc2 = nn.Linear(1000, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         # x = self.drop(x)
#         if x.size(0) > 1:
#             x = self.bn1(x)
#         x = torch.sigmoid(self.fc2(x))
#         return x
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model = RiverNDMLP(4).to(device)
#
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
#             predictions = (outputs >= 0.5).float()
#             correct += (predictions == batch_y).float().sum().item()
#             total += batch_y.size(0)
#     return correct / total
#
#
# train_accuracies = []
# val_accuracies = []
# test_accuracies = []
#
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#     train_accuracy = evaluate_model(model, train_loader)
#     # val_accuracy = evaluate_model(model, val_loader)
#     test_accuracy = evaluate_model(model, test_loader)
#
#     train_accuracies.append(train_accuracy)
#     # val_accuracies.append(val_accuracy)
#     test_accuracies.append(test_accuracy)
#
#     print(f'Epoch {epoch + 1}/{num_epochs} - Train Accuracy: {train_accuracy * 100:.2f}%, '
#           # f'Val Accuracy: {val_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')
#           f'No val accuracy, Test Accuracy: {test_accuracy * 100:.2f}%')
#
# torch.save(model, '../FrozenModels/30_samples/30_centralized_model_no_val_red_2_5')
#
# epochs = range(1, num_epochs + 1)
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, train_accuracies, label='Train Accuracy')
# # plt.plot(epochs, val_accuracies, label='Validation Accuracy')
# plt.plot(epochs, test_accuracies, label='Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training, Validation, and Test Accuracy Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()


# 93.75 centralized 100 data points 70 ep
# 93.3 federated iid 100 dp 20 clients 2 per round 50 ep/10 l_ep
# 93.3 federated non-iid 100 dp 20 clients 2 per round 50 ep/10 l_ep

# 94 centralized 100 dp 130 ep
# 96 federated iid 100 dp 20 clients 2 per round 50 ep/10 l_ep
# 94 federated non-iid 100 dp 20 clients 2 per round 50 ep/10 l_ep

# 93 centralized 100 dp - 56 test - 130 ep
# 93 federaed iid 100 dp - 56 test - 20 clients 2 per round 50 ep/10 l_ep
# 95 federated non-iid 100 dp - 52 test - 20 clients 2 per round 50 ep/10 l_ep

