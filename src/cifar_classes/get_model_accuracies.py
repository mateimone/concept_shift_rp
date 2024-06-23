import torch
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm


def load_model(filepath):
    model = torch.load(filepath).to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader):
            batch_y = batch_y.to(device)
            batch_X = F.interpolate(batch_X, size=(224, 224), mode='bilinear', align_corners=False).to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    return accuracy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = v2.Compose([
    v2.ColorJitter(brightness=1, contrast=0.9),
    v2.Grayscale(num_output_channels=3),
    v2.RandomHorizontalFlip(p=0.5),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
    v2.ToTensor(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


test_set = datasets.CIFAR10(root='../../data/cifar', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

model_iid = load_model('../FrozenModels/cifar_color_jitter/iid_train_nor_0cmodified_1')
model_non_iid = load_model('../FrozenModels/cifar_color_jitter/dirich_nottr_1')
model_centralized = load_model('../FrozenModels/cifar_color_jitter/centralized_train_nor_0cmodified_real')
model_iid_shift = load_model('../FrozenModels/cifar_color_jitter/iid_train_mix_2cmodified_1')
model_non_iid_shift = load_model('../FrozenModels/cifar_color_jitter/dirich_tr_1')
model_centralized_shift = load_model('../FrozenModels/cifar_color_jitter/20perc_centralized_train_mix_test_mix_0cmodified')

accuracy_iid = evaluate_model(model_iid, test_loader)
accuracy_non_iid = evaluate_model(model_non_iid, test_loader)
accuracy_centralized = evaluate_model(model_centralized, test_loader)
accuracy_iid_shift = evaluate_model(model_iid_shift, test_loader)
accuracy_non_iid_shift = evaluate_model(model_non_iid_shift, test_loader)
accuracy_centralized_shift = evaluate_model(model_centralized_shift, test_loader)

print(f'Accuracy of Federated IID model: {accuracy_iid:.2f}%')
print(f'Accuracy of Federated Non-IID model: {accuracy_non_iid:.2f}%')
print(f'Accuracy of Centralized model: {accuracy_centralized:.2f}%')
print(f'Accuracy of Federated IID model, shift: {accuracy_iid_shift:.2f}%')
print(f'Accuracy of Federated Non-IID model, shift: {accuracy_non_iid_shift:.2f}%')
print(f'Accuracy of Centralized model, shift: {accuracy_centralized_shift:.2f}%')
