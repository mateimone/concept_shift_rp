import csv
import numpy as np
import matplotlib.pyplot as plt
from river.datasets import synth
from sklearn.svm import SVC


def load_data_from_csv(file_path):
    features = []
    labels = []
    redundant_labels = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            features.append([float(row[0]), float(row[1])])
            labels.append(int(row[len(row)-1]))
            redundant_labels.append([float(row[i+2]) for i in range(len(row)-3)])
    return np.array(features), np.array(labels), np.array(redundant_labels)


def get_line_params(w, b):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    return slope, intercept


def get_midpoint(slope, intercept):
    x1, y1 = 0, intercept
    x2, y2 = 1, slope + intercept
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    return midpoint_x, midpoint_y


def rotate_line_around_center(slope, intercept, midpoint_x, midpoint_y, degrees):
    theta = np.radians(degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x1, y1 = 0, intercept
    x2, y2 = 1, slope + intercept

    x1_translated, y1_translated = x1 - midpoint_x, y1 - midpoint_y
    x2_translated, y2_translated = x2 - midpoint_x, y2 - midpoint_y

    x1_rot = x1_translated * cos_theta - y1_translated * sin_theta
    y1_rot = x1_translated * sin_theta + y1_translated * cos_theta
    x2_rot = x2_translated * cos_theta - y2_translated * sin_theta
    y2_rot = x2_translated * sin_theta + y2_translated * cos_theta

    x1_final, y1_final = x1_rot + midpoint_x, y1_rot + midpoint_y
    x2_final, y2_final = x2_rot + midpoint_x, y2_rot + midpoint_y

    new_slope = (y2_final - y1_final) / (x2_final - x1_final)
    new_intercept = y1_final - new_slope * x1_final

    return new_slope, new_intercept


def classify_points(features, slope, intercept):
    return (features[:, 1] > slope * features[:, 0] + intercept).astype(int)


def plot_dataset(features, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.colorbar(label='Label')
    plt.grid(True)
    plt.show()

def func(rotation_degrees: int):
    csv_file = "../hyperplanes/10k_samples_hyperplane/10k_boundary_rotated_0_hyperplane.csv"
    original_features, original_labels, redundant_features = load_data_from_csv(csv_file)

    svm = SVC(kernel='linear')
    svm.fit(original_features, original_labels)

    w = svm.coef_[0]
    b = svm.intercept_[0]

    slope, intercept = get_line_params(w, b)

    midpoint_x, midpoint_y = get_midpoint(slope, intercept)

    rotated_slope, rotated_intercept = rotate_line_around_center(slope, intercept, midpoint_x, midpoint_y, rotation_degrees)

    original_classifications = classify_points(original_features, slope, intercept)
    noise_indices = np.where(original_classifications != original_labels)[0]
    rotated_labels = classify_points(original_features, rotated_slope, rotated_intercept)

    if 135 <= rotation_degrees <= 315:
        rotated_labels = 1 - rotated_labels

    original_file = '../hyperplanes/200k_samples_hyperplane/boundary_rotated_0_hyperplane.csv'
    with open(original_file, mode='w', newline='') as ori_file:
        rot_writer = csv.writer(ori_file)
        header = ['feature1', 'feature2'] + [f'redundant_feature_{i+1}' for i in range(redundant_features.shape[1])] + ['label']
        # header = ['feature1', 'feature2'] + ['label']
        rot_writer.writerow(header)

        for feature, redundant, label in zip(original_features, redundant_features, rotated_labels):
        # for feature, label in zip(original_features, original_labels):
            row = list(feature) + list(redundant) + [label]
            # row = list(feature) + [label]
            rot_writer.writerow(row)

    # rotated_file = f'../hyperplanes_with_redundancy/200k_samples_hyperplane_red_5/200k_boundary_rotated_{rotation_degrees}_hyperplane.csv'
    # with open(rotated_file, mode='w', newline='') as rot_file:
    #     rot_writer = csv.writer(rot_file)
    #     header = ['feature1', 'feature2'] + [f'redundant_feature_{i+1}' for i in range(redundant_features.shape[1])] + ['label']
    #     # header = ['feature1', 'feature2'] + ['label']
    #     rot_writer.writerow(header)
    #
    #     for feature, redundant, label in zip(original_features, redundant_features, rotated_labels):
    #     # for feature, label in zip(original_features, rotated_labels):
    #         row = list(feature) + list(redundant) + [label]
    #         # row = list(feature) + [label]
    #         rot_writer.writerow(row)
    #
    # print("Boundary rotated dataset generated and saved.")

    plot_dataset(original_features, original_labels, 'Original Hyperplane')
    plot_dataset(original_features, rotated_labels, 'Boundary Rotated Hyperplane')

func(0)
func(10)
func(20)
func(30)
func(40)
func(50)
func(60)
func(70)
func(80)
func(90)

