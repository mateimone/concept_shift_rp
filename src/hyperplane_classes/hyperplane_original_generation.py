import csv
from river.datasets import synth


dataset = synth.Hyperplane(seed=2, n_features=2)
# dataset = synth.Sine(seed=2, balance_classes=False, has_noise=False)

with open('../hyperplanes/1k_samples_hyperplane/1k_boundary_rotated_0_hyperplane.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    header = [f'feature{i+1}' for i in range(2)] + ['label']
    writer.writerow(header)

    for x, y in dataset.take(1000):
        row = [x[i] for i in range(2)] + [y]
        writer.writerow(row)
