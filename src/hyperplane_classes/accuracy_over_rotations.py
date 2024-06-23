import matplotlib.pyplot as plt
from src.hyperplane_classes.non_torch_datasets import RiverNDDataset
import torch
from torch.utils.data import DataLoader
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

rotation_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
file_paths = [f"../hyperplanes_with_redundancy/200k_samples_hyperplane_red_5/200k_boundary_rotated_{deg}_hyperplane.csv" for deg in rotation_degrees]

def load_models(model_paths):
    models = [torch.load(path).to(device) for path in model_paths]
    return models

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

def get_average_accuracies(model_paths):
    models = load_models(model_paths)
    accuracies = {deg: [] for deg in rotation_degrees}
    print(model_paths[0])
    i = 0
    for model in models:
        print("model" + str(i))
        for file_path, deg in zip(file_paths, rotation_degrees):
            test_dataset = RiverNDDataset(file_path, 7)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            accuracy = evaluate_model(model, test_loader)
            print(f"{deg}, accuracy: {accuracy}")
            accuracies[deg].append(accuracy)

    average_accuracies = [np.mean(accuracies[deg]) for deg in rotation_degrees]
    std_deviations = [np.std(accuracies[deg]) for deg in rotation_degrees]
    i += 1
    return average_accuracies, std_deviations

centralized_model_paths = [f"../FrozenModels/30_samples/30_centralized_model_no_val_red_5_{i}" for i in range(1, 6)]
iid_model_paths = [f"../FrozenModels/30_samples/30_iid_model_no_val_red_5_{i}" for i in range(6, 11)]
dirichlet_model_paths = [f"../FrozenModels/30_samples/30_dirichlet_model_no_val_red_5_{i}" for i in range(1, 6)]

centralized_accuracies, centralized_std = get_average_accuracies(centralized_model_paths)
iid_accuracies, iid_std = get_average_accuracies(iid_model_paths)
dirichlet_accuracies, dirichlet_std = get_average_accuracies(dirichlet_model_paths)

plt.figure(figsize=(10, 5))
plt.errorbar(rotation_degrees, centralized_accuracies, yerr=centralized_std, marker='o', label='Centralized Model', capsize=5, alpha=0.6)
plt.errorbar(rotation_degrees, iid_accuracies, yerr=iid_std, marker='x', label='IID Model', capsize=5, alpha=0.8)
plt.errorbar(rotation_degrees, dirichlet_accuracies, yerr=dirichlet_std, marker='s', label='Non-IID Model', capsize=5, alpha=1)
plt.xlabel('Rotation Degrees')
plt.ylabel('Accuracy')
plt.title('Average Accuracy over Rotation Degrees with 30 Training Samples on 200k Test Samples, 5 Redundant Features')
plt.grid(True)
plt.legend()
plt.show()
