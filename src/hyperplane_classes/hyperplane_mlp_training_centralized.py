import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
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

def main(model_index, degrees):
    full_train_dataset = RiverNDDataset(f"../hyperplanes/100_samples_hyperplane/100_boundary_rotated_{degrees}_hyperplane.csv", 2)
    test_dataset = RiverNDDataset(
        "../hyperplanes/100_samples_hyperplane/100_boundary_rotated_0_hyperplane.csv", 2)

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RiverNDMLP(2).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_size = len(full_train_dataset)
    train_dataset = full_train_dataset
    # train_size = int(0.8 * len(full_train_dataset))
    # val_size = len(full_train_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        train_accuracy = evaluate_model(model, train_loader)
        # val_accuracy = evaluate_model(model, val_loader)
        test_accuracy = evaluate_model(model, test_loader)

        train_accuracies.append(train_accuracy)
        # val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Accuracy: {train_accuracy * 100:.2f}%, '
              # f'Val Accuracy: {val_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')
              f'No val accuracy, Test Accuracy: {test_accuracy * 100:.2f}%')

    torch.save(model, f'../FrozenModels/plotting_purposes/10k_centralized_model_no_val_{degrees}')

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    # plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation, and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

main(1, 0)
main(1, 10)
main(1, 20)
main(1, 30)
main(1, 40)
main(1, 50)
main(1, 60)
main(1, 70)
main(1, 80)
main(1, 90)
