import pickle

import numpy as np

with open("training/models/all-MiniLM-L6-v2/final/centroids.pkl", "rb") as f:
    centroids_dict = pickle.load(f)

print("Centroid norms:")
for cat, emb in centroids_dict.items():
    norm = np.linalg.norm(emb)
    print(f"  {cat}: {norm:.4f}")

# Check if they need normalization
print("\nShould be ~1.0 if normalized")
