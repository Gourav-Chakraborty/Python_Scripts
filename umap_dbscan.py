import mdtraj as md
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import sys

# Load MD trajectory using mdtraj
trajectory = md.load("umap.nc", top="umap.prmtop")
print("Trajectories loaded")

# Extract features (atomic coordinates)
features = trajectory.xyz.reshape(trajectory.n_frames, -1)
# Scale the data once
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
print("Total frames:", len(features))
print("Features shape:", features.shape)


# Define a function to evaluate a single combination of UMAP parameters
def evaluate_umap_params(n_neighbors, min_dist, data):
    print(f"Running UMAP with min_dist={min_dist} and n_neighbors={n_neighbors}")
    # Create and fit UMAP model with n_jobs=2 to avoid Numba thread issue
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=2)
    umap_embedding = umap_model.fit_transform(data)
    # Perform clustering with DBSCAN
    dbscan = DBSCAN()
    cluster_labels = dbscan.fit_predict(umap_embedding)
    # Calculate silhouette score
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        silhouette = -1
    else:
        silhouette = silhouette_score(umap_embedding, cluster_labels)
    return silhouette, [n_neighbors, min_dist]
# Function to tune UMAP hyperparameters and save silhouette scores
def tune_umap_silhouette(data, max_neighbors=2500, min_dist_start=0.1, min_dist_end=0.8, min_dist_stride=0.2):
    silhouette_scores = []
    param_combinations = []
    # Create range of min_dist values
    min_dist_range = np.arange(min_dist_start, min_dist_end + min_dist_stride, min_dist_stride)
    # Define progress bar setting based on the environment
    use_progress_bar = not sys.stdout.isatty()
    # Use joblib to parallelize the UMAP tuning
    results = Parallel(n_jobs=16)(delayed(evaluate_umap_params)(n, m, data)
                                  for m in tqdm(min_dist_range, desc="UMAP Hyperparameters", disable=not use_progress_bar)
                                  for n in range(10, max_neighbors + 1, 50))

    # Collect results
    silhouette_scores = [result[0] for result in results]
    param_combinations = [result[1] for result in results]
    # Find best parameters based on highest silhouette score
    best_idx = np.argmax(silhouette_scores)  # Find index with highest score
    best_params = param_combinations[best_idx]
    print("Best UMAP Hyperparameters based on Silhouette Score:")
    print("n_neighbors:", best_params[0])
    print("min_dist:", best_params[1])
    # Save silhouette scores for each hyperparameter combination
    df_scores = pd.DataFrame({
        'n_neighbors': [p[0] for p in param_combinations],
        'min_dist': [p[1] for p in param_combinations],
        'silhouette_score': silhouette_scores
    })
    df_scores.to_csv('umap_silhouette_scores.csv', index=False)
    return best_params, max(silhouette_scores)  # Return best params and best silhouette score
# Perform hyperparameter tuning for UMAP
best_params_umap, best_silhouette_score = tune_umap_silhouette(scaled_data)
# Create UMAP model with best hyperparameters and set n_jobs=2
umap_model = UMAP(n_components=2, n_neighbors=best_params_umap[0], min_dist=best_params_umap[1], n_jobs=2)
# Generate UMAP embedding
umap_embedding = umap_model.fit_transform(scaled_data)
# Create a DataFrame with UMAP-1, UMAP-2, and frame numbers
umap_df = pd.DataFrame({
    'UMAP_1': umap_embedding[:, 0],
    'UMAP_2': umap_embedding[:, 1],
    'Frame_Number': np.arange(len(trajectory))  # Use arange for correct frame numbers
})
# Save the DataFrame to a CSV file
umap_df.to_csv('umap_data_with_frame_numbers.csv', index=False)
print("UMAP embedding generated and saved to CSV.")
print("Best Silhouette Score:", best_silhouette_score)
