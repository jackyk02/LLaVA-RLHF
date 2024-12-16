from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Load the uploaded Excel file
file_path = 'octo_data.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Sheet1')
filtered_df = df

# Function to calculate Mean Squared Error (nrmse)
min_values = np.array([-0.07631552, -0.15209596, -0.15171302, -
                      0.22191394, -0.34210532, -0.73485305,  0.00000000])
max_values = np.array([0.08137930,  0.14595977,  0.14885315,
                      0.22793450,  0.20718527,  0.78006949,  1.00000000])
ranges = max_values - min_values


def calculate_nrmse(row):
    l0 = np.fromstring(row['action0'].strip('[]').replace(
        '\n', '').replace('  ', ' '), sep=' ')
    l1 = np.fromstring(row['action1'].strip('[]').replace(
        '\n', '').replace('  ', ' '), sep=' ')

    # Normalize the difference by the range
    normalized_diff = (l0 - l1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse


# Add the nrmse column
print("Calculating nrmse for each row...")
filtered_df['nrmse'] = filtered_df.apply(calculate_nrmse, axis=1)
filtered_df


def string_to_array(action_str):
    return np.fromstring(action_str.strip('[]').replace('\n', '').replace(' ', ' '), sep=' ')


def generate_comparisons(df):
    rows = []

    # Get unique indices for progress tracking
    unique_indices = df['index'].unique()

    print(f"\nProcessing {len(unique_indices)} unique indices...")
    for idx in tqdm(unique_indices, desc="Processing groups"):
        group = df[df['index'] == idx]

        # Convert action1 strings to arrays for clustering
        action_arrays = np.array([string_to_array(action)
                                  for action in group['action1']])

        # Apply k-means clustering
        # Ensure we don't try to create more clusters than samples
        n_clusters = min(9, len(group))
        if n_clusters < 2:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(action_arrays)

        # Create a dictionary to store one representative sample from each cluster
        cluster_representatives = {}
        for i, (_, row) in enumerate(group.iterrows()):
            cluster = cluster_labels[i]
            if cluster not in cluster_representatives:
                cluster_representatives[cluster] = row

        # Generate combinations of cluster representatives
        rep_combinations = list(combinations(
            cluster_representatives.values(), 2))

        count = 1
        for row1, row2 in rep_combinations:
            # Determine the winner based on the lower nrmse
            if count == 33:
                break
            if row1['nrmse'] == row2['nrmse']:
                continue

            winner = 1 if row1['nrmse'] < row2['nrmse'] else 2

            rows.append({
                'index': str(row1['index']),
                'grountruth': str(row1['action0']),
                'action0': str(row1['action1']),
                'action1': str(row2['action1']),
                'pair_index': str(count),
                'pair1_nrmse': str(row1['nrmse']),
                'pair2_nrmse': str(row2['nrmse']),
                'winner': winner
            })
            count += 1

    return pd.DataFrame(rows)


print("\nGenerating comparisons...")
comparison_df = generate_comparisons(filtered_df)
print("\nSaving results to Excel...")
comparison_df.to_excel('k_means_data_octo.xlsx', index=False)
print("Done!")
