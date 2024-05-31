import pandas as pd
import matplotlib.pyplot as plt


def plot():
    df = pd.read_csv('../hyperplanes/abrupt_data_H_0.csv')
    print(sum(df['label'] == 0), sum(df['label'] == 1))

    if 'feature1' not in df.columns or 'feature2' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'feature1', 'feature2', and 'label' columns")

    x1 = df['feature1']
    x2 = df['feature2']
    labels = df['label']
    i = 0
    plt.figure(figsize=(8, 6))
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(subset['feature1'], subset['feature2'], label=f'Class {label}', alpha=0.7)
        break

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Points with Labels')
    plt.legend()
    plt.grid(True)
    plt.show()
