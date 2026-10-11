import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
train = pd.DataFrame(iris.data, columns=iris.feature_names)

# Simulation de nouvelles données de production :
# les pétales sont plus grands et les mesures un peu plus bruitées
rng = np.random.default_rng(42)
prod = train.copy()
prod["petal length (cm)"] = prod["petal length (cm)"] * 1.3 + rng.normal(0, 0.2, len(prod))
prod["petal width (cm)"] = prod["petal width (cm)"] * 1.3 + rng.normal(0, 0.1, len(prod))

print("=== Entrainement ===")
print(train.describe().loc[["mean", "std", "min", "max"]].round(2))
print("\n=== Production (simulée) ===")
print(prod.describe().loc[["mean", "std", "min", "max"]].round(2))

# Alerte simple : écart de moyenne > 10 %
print("\n=== Détection de drift ===")
for col in train.columns:
    ecart = abs(prod[col].mean() - train[col].mean()) / train[col].mean() * 100
    statut = "ALERTE DRIFT" if ecart > 10 else "OK"
    print(f"{col:22s} écart moyenne : {ecart:5.1f} %  -> {statut}")