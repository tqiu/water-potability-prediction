import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
# test_data = pd.read_csv("data/processed/test_processed_data.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop("Potability", axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
# X_test = test_data.drop("Potability", axis=1)
# y_test = test_data["Potability"]

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")
# model = pickle.load(open("model.pkl", "rb"))

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_dict = {"accuracy": accuracy, "f1_score": f1}
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")    
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1_score = f1_score(y_test, y_pred)

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    try:
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")

def main():
    try:
        test_data_filepath = "data/processed/test_processed_data.csv"
        model_filepath = "models/model.pkl"
        metrics_filepath = "reports/metrics.json"

        test_data = load_data(test_data_filepath)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_filepath)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_filepath)
    except Exception as e:
        raise Exception(f"Error in model evaluation: {e}")
    

if __name__ == "__main__":
    main()
