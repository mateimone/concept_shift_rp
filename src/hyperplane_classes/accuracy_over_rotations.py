# import matplotlib.pyplot as plt
# from src.models import RiverNDMLP
# from src.non_torch_datasets import RiverNDDataset
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
#
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Define the rotation degrees and corresponding file paths
# rotation_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# file_paths = [f"../hyperplanes/200k_samples_hyperplane/boundary_rotated_{deg}_hyperplane.csv" for deg in rotation_degrees]
#
# # Function to load multiple models and average their accuracies, and compute standard deviation
# def load_models(model_paths):
#     models = [torch.load(path).to(device) for path in model_paths]
#     return models
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
# def get_average_accuracies(model_paths):
#     models = load_models(model_paths)
#     accuracies = {deg: [] for deg in rotation_degrees}
#
#     for model in models:
#         for file_path, deg in zip(file_paths, rotation_degrees):
#             test_dataset = RiverNDDataset(file_path, 2)
#             test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
#             accuracy = evaluate_model(model, test_loader)
#             accuracies[deg].append(accuracy)
#
#     average_accuracies = [np.mean(accuracies[deg]) for deg in rotation_degrees]
#     std_deviations = [np.std(accuracies[deg]) for deg in rotation_degrees]
#     return average_accuracies, std_deviations
#
#
# # Paths to your saved models (5 models for each type)
# centralized_model_paths = [f"../FrozenModels/30_samples/30_centralized_model_no_val_{i}" for i in range(1, 6)]
# iid_model_paths = [f"../FrozenModels/30_samples/30_iid_model_no_val_{i}" for i in range(1, 6)]
# dirichlet_model_paths = [f"../FrozenModels/30_samples/30_dirichlet_model_no_val_{i}" for i in range(1, 6)]
#
# # Get average accuracies and standard deviations for each model type
# centralized_accuracies, centralized_std = get_average_accuracies(centralized_model_paths)
# iid_accuracies, iid_std = get_average_accuracies(iid_model_paths)
# dirichlet_accuracies, dirichlet_std = get_average_accuracies(dirichlet_model_paths)
#
# # Plotting
# plt.figure(figsize=(10, 5))
#
# # Plot each model type with confidence interval bands
# for accuracies, std_dev, label, color in zip(
#     [centralized_accuracies, iid_accuracies, dirichlet_accuracies],
#     [centralized_std, iid_std, dirichlet_std],
#     ['Centralized Model', 'IID Model', 'Non-IID Model'],
#     ['blue', 'orange', 'green']):
#
#     plt.fill_between(rotation_degrees,
#                      np.array(accuracies) - np.array(std_dev),
#                      np.array(accuracies) + np.array(std_dev),
#                      color=color, alpha=0.1)
#     plt.plot(rotation_degrees, accuracies, marker='o', label=label, color=color)
#
# plt.xlabel('Rotation Degrees')
# plt.ylabel('Accuracy')
# plt.title('Average Accuracy and Variance over Rotation Degrees')
# plt.grid(True)
# plt.legend()
# plt.show()


# import matplotlib.pyplot as plt
# from src.models import RiverNDMLP
# from src.non_torch_datasets import RiverNDDataset
# import torch
# from torch.utils.data import DataLoader
# import numpy as np  # Import numpy for standard deviation calculation
#
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Define the rotation degrees and corresponding file paths
# rotation_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# file_paths = [f"../hyperplanes/200k_samples_hyperplane/boundary_rotated_{deg}_hyperplane.csv" for deg in rotation_degrees]
#
# # Function to load multiple models and average their accuracies, and compute standard deviation
# def load_models(model_paths):
#     models = [torch.load(path).to(device) for path in model_paths]
#     return models
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
# def get_average_accuracies(model_paths):
#     models = load_models(model_paths)
#     accuracies = {deg: [] for deg in rotation_degrees}
#
#     for model in models:
#         for file_path, deg in zip(file_paths, rotation_degrees):
#             test_dataset = RiverNDDataset(file_path, 2)
#             test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
#             accuracy = evaluate_model(model, test_loader)
#             accuracies[deg].append(accuracy)
#
#     average_accuracies = [np.mean(accuracies[deg]) for deg in rotation_degrees]
#     std_deviations = [np.std(accuracies[deg]) for deg in rotation_degrees]
#     return average_accuracies, std_deviations
#
# # Paths to your saved models (5 models for each type)
# centralized_model_paths = [f"../FrozenModels/30_samples/30_centralized_model_no_val_{i}" for i in range(1, 6)]
# iid_model_paths = [f"../FrozenModels/30_samples/30_iid_model_no_val_{i}" for i in range(1, 6)]
# dirichlet_model_paths = [f"../FrozenModels/30_samples/30_dirichlet_model_no_val_{i}" for i in range(1, 6)]
#
# # Get average accuracies and standard deviations for each model type
# centralized_accuracies, centralized_std = get_average_accuracies(centralized_model_paths)
# iid_accuracies, iid_std = get_average_accuracies(iid_model_paths)
# dirichlet_accuracies, dirichlet_std = get_average_accuracies(dirichlet_model_paths)
#
# # Plot the results with error bars
# plt.figure(figsize=(10, 5))
# plt.errorbar(rotation_degrees, centralized_accuracies, yerr=centralized_std, marker='o', label='Centralized Model', capsize=5)
# plt.errorbar(rotation_degrees, iid_accuracies, yerr=iid_std, marker='x', label='IID Model', capsize=5)
# plt.errorbar(rotation_degrees, dirichlet_accuracies, yerr=dirichlet_std, marker='s', label='Non-IID Model', capsize=5)
# plt.xlabel('Rotation Degrees')
# plt.ylabel('Accuracy')
# plt.title('Avg. Accuracy over Rotation Degrees with 30 Training Samples on 200k Test Samples, 2 Redundant Features')
# plt.grid(True)
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
from src.non_torch_datasets import RiverNDDataset
import torch
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the rotation degrees and corresponding file paths
rotation_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
file_paths = [f"../hyperplanes/200k_samples_hyperplane/boundary_rotated_{deg}_hyperplane.csv" for deg in rotation_degrees]
# file_paths.insert(0, "../hyperplanes/200k_samples_hyperplane_red_5/boundary_rotated_0_hyperplane.csv")

# Function to load multiple models and average their accuracies
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

    for model in models:
        for file_path, deg in zip(file_paths, rotation_degrees):
            test_dataset = RiverNDDataset(file_path, 2)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            accuracy = evaluate_model(model, test_loader)
            accuracies[deg].append(accuracy)

    average_accuracies = [sum(accuracies[deg]) / len(accuracies[deg]) for deg in rotation_degrees]
    return average_accuracies

# Paths to your saved models (5 models for each type)
centralized_model_paths = [f"../FrozenModels/30_samples/30_centralized_model_no_val_{i}" for i in range(1, 6)]
iid_model_paths = [f"../FrozenModels/30_samples/30_iid_model_no_val_{i}" for i in range(1, 6)]
dirichlet_model_paths = [f"../FrozenModels/30_samples/30_dirichlet_model_no_val_{i}" for i in range(1, 6)]

# Get average accuracies for each model type
centralized_accuracies = get_average_accuracies(centralized_model_paths)
iid_accuracies = get_average_accuracies(iid_model_paths)
dirichlet_accuracies = get_average_accuracies(dirichlet_model_paths)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(rotation_degrees, centralized_accuracies, marker='o', label='Centralized Model')
plt.plot(rotation_degrees, iid_accuracies, marker='x', label='IID Model')
plt.plot(rotation_degrees, dirichlet_accuracies, marker='s', label='Non-IID Model')
plt.xlabel('Rotation Degrees')
plt.ylabel('Accuracy')
plt.title('Average Accuracy over Rotation Degrees with 30 Training Samples on 200k Test Samples')
plt.grid(True)
plt.legend()
plt.show()

