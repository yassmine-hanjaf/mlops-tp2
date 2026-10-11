import joblib
from sklearn.datasets import load_iris

model = joblib.load("models/iris_model.pkl")
iris = load_iris()

p = model.predict([iris.data[0]])
print(p[0], iris.target_names[p[0]])