import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Mock data
X = np.random.rand(3000, 20)
y = np.random.randint(0, 5, size=(3000,))

print("Benchmarking n_jobs=-1...")
t0 = time.time()
for _ in range(5):
    cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        X, y, cv=StratifiedKFold(3), scoring='accuracy'
    )
print(f"n_jobs=-1 took: {time.time() - t0:.2f} seconds for 5 runs")

print("Benchmarking n_jobs=1...")
t0 = time.time()
for _ in range(5):
    cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        X, y, cv=StratifiedKFold(3), scoring='accuracy'
    )
print(f"n_jobs=1 took: {time.time() - t0:.2f} seconds for 5 runs")
