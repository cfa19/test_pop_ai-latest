import pickle

import numpy as np

with open("training/models/all-MiniLM-L6-v2/final/centroids.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
if hasattr(data, "shape"):
    print("Shape:", data.shape)
if isinstance(data, dict):
    print("Keys:", list(data.keys())[:5])
    for k, v in list(data.items())[:2]:
        print(f"  {k}: {type(v)}, shape={getattr(v, 'shape', None)}")
elif isinstance(data, (list, np.ndarray)):
    arr = np.array(data)
    print("As array shape:", arr.shape)
    print("First row norm:", np.linalg.norm(arr[0]))
    print("First 5 values:", arr[0][:5])
