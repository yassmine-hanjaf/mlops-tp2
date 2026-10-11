import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Iris_MLOps")

iris = load_iris()
X, y = iris.data, iris.target
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model = None
best_acc = -1

for n in [50, 100, 200]:
    with mlflow.start_run(run_name=f"rf_{n}"):
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))

        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"n_estimators = {n} -> Accuracy : {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/iris_model.pkl")
print("Modèle sauvegardé. Meilleure accuracy :", best_acc)