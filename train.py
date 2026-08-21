import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Load dataset
data = pd.read_csv("data/iris.csv")

# Features and target
X = data.drop(["species", "dataset_version"], axis=1)
y = data["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# Train model
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    random_state=params["random_state"]
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save metrics
metrics = {
    "accuracy": accuracy
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)