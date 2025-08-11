import json
import numpy as np
from sklearn.cluster import KMeans
import os

# Get the directory where the script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the coin_flow_data.json file in the data folder.
file_path = os.path.join(script_dir, '..', 'data', 'coin_flow_data.json')

# Load the aggregated coin flow data from the JSON file.
with open(file_path, "r") as f:
    data = json.load(f)

aggregated_flows = data["aggregated_flows"]

# Prepare the data: create lists of coin names and their corresponding net flows.
coins = list(aggregated_flows.keys())
flows = np.array([aggregated_flows[coin] for coin in coins]).reshape(-1, 1)

# Use KMeans clustering to divide the coins into two groups.
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(flows)

# Calculate the average flow for each cluster.
cluster_averages = {}
for cluster in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    avg_flow = flows[cluster_indices].mean()
    cluster_averages[cluster] = avg_flow

# Identify the cluster with the higher average net flow as the "Good" cluster.
good_cluster = max(cluster_averages, key=cluster_averages.get)

# Create a classification dictionary based on the AI model (KMeans).
coin_classification_ai = {}
for i, coin in enumerate(coins):
    if clusters[i] == good_cluster:
        coin_classification_ai[coin] = "Good Coin (AI Model)"
    else:
        coin_classification_ai[coin] = "Bad Coin (AI Model)"

# Save the AI-based classification results to a JSON file in the data folder.
output_file_path = os.path.join(script_dir, '..', 'data', 'coin_classification_ai.json')
with open(output_file_path, "w") as outfile:
    json.dump(coin_classification_ai, outfile, indent=4)

print("AI-based classification results saved to", output_file_path)
