import mlflow
import joblib
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['target'])

# Visualize data
scatter_matrix(X, figsize=[12, 8], alpha=0.6, c=y["target"], cmap=plt.get_cmap("jet"))
plt.savefig("iris_data_plot.png")

# Train/test split
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
knc = KNeighborsClassifier()
knc.fit(X_train_set, y_train_set.values.ravel())
y_pred = knc.predict(X_test_set)

# Evaluation
cm = confusion_matrix(y_test_set, y_pred)
print(classification_report(y_test_set, y_pred))

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "KNeighborsClassifier")
    mlflow.log_metric("accuracy", knc.score(X_test_set, y_test_set))
    mlflow.sklearn.log_model(knc, "model")
    mlflow.log_artifact("iris_data_plot.png")

joblib.dump(knc, "model.pkl")