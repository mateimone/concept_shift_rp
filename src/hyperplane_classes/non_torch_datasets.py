from torch.utils.data.dataset import Dataset
import pandas as pd
import torch


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