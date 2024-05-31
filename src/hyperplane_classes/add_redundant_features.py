import pandas as pd
import numpy as np

def add_redundancy(rot: int):
    # Read the original CSV file
    file_path = f"../hyperplanes/30_samples_hyperplane/30_boundary_rotated_{rot}_hyperplane.csv"
    df = pd.read_csv(file_path)
    original_labels = df['label'].copy()

    print(df.head())

    num_redundant_features = 2

    for i in range(num_redundant_features):
        redundant_feature = np.random.normal(size=len(df))
        redundant_feature = (redundant_feature - redundant_feature.min()) / (redundant_feature.max() - redundant_feature.min())
        df[f"redundant_feature_{i+1}"] = redundant_feature

    # Move the label column to the end
    label_column = df.pop("label")
    df["label"] = label_column

    updated_file_path = f"../hyperplanes_with_redundancy/30_samples_hyperplane_red_2/30_boundary_rotated_{rot}_hyperplane.csv"
    df.to_csv(updated_file_path, index=False)

    new_df = pd.read_csv(updated_file_path)

    assert new_df['label'].equals(original_labels)

    print(df.head())

add_redundancy(0)
# add_redundancy(10)
# add_redundancy(20)
# add_redundancy(30)
# add_redundancy(40)
# add_redundancy(50)
# add_redundancy(60)
# add_redundancy(70)
# add_redundancy(80)
# add_redundancy(90)

