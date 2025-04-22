import flwr as fl
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import log_loss, confusion_matrix, classification_report
from utils import load_client_data, train_xgboost, train_logistic, train_catboost
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# -------------------------------
# Command-line args
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, default=0, help="Client ID")
parser.add_argument("--model_type", type=str, choices=["xgb", "catboost", "lr"], default="xgb", help="Model type")
args = parser.parse_args()

client_id = args.client_id
model_type = args.model_type
X_train, y_train, X_test, y_test = load_client_data(client_id)

os.makedirs("plots", exist_ok=True)

# -------------------------------
# Unified Flower Client
# -------------------------------
class UnifiedClient(fl.client.NumPyClient):
    def __init__(self):
        self.model_path = f"model_{model_type}_client{client_id}.{'bst' if model_type == 'xgb' else 'pkl'}"
        self.model = self._train_base_model()
        self._save_model()

    def _train_base_model(self):
        print(f"🚀 [Client {client_id}] Training {model_type.upper()} model...")

        if model_type == "xgb":
            model = train_xgboost(X_train, y_train)
        elif model_type == "catboost":
            model = train_catboost(X_train, y_train)
        else:
            model = train_logistic(X_train, y_train)

        # Plot training loss (manual for catboost/lr)
        self._plot_training_loss(model)
        return model

    def _plot_training_loss(self, model):
        plt.figure(figsize=(6, 4))
        if model_type == "xgb":
            logloss = model.attributes().get("best_score", [])
            if hasattr(model, 'evals_result'):
                losses = model.evals_result()["train"]["logloss"]
                plt.plot(losses)
        elif model_type == "catboost":
            losses = model.get_evals_result()["learn"]["Logloss"]
            plt.plot(losses)
        else:  # Logistic regression (track loss on training set)
            proba = model.predict_proba(X_train)
            losses = [log_loss(y_train, proba)]
            plt.plot(losses)

        plt.title(f"{model_type.upper()} Loss (Client {client_id})")
        plt.xlabel("Epochs" if model_type != "lr" else "Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/loss_{model_type}_client{client_id}.png")
        plt.close()

    def _save_model(self):
        if model_type == "xgb":
            self.model.save_model(self.model_path)
        else:
            joblib.dump(self.model, self.model_path)

    def _load_model(self):
        if model_type == "xgb":
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
        else:
            self.model = joblib.load(self.model_path)

    def get_parameters(self, config):
        return [np.array([0.0])]  # Placeholder

    def set_parameters(self, parameters):
        try:
            self._load_model()
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def _predict(self, X):
        if model_type == "xgb":
            return (self.model.predict(xgb.DMatrix(X)) > 0.5).astype(int)
        return self.model.predict(X).astype(int)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = self._train_base_model()
        self._save_model()
        preds = self._predict(X_train)
        acc = np.mean(preds == y_train)
        print(f"✅ Fit complete — Accuracy: {acc:.4f}")
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self._predict(X_test)
        acc = np.mean(preds == y_test)
        print(f"🔍 Eval — Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        plt.title(f"Confusion Matrix (Client {client_id})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix_{model_type}_client{client_id}.png")
        plt.close()

        report = classification_report(y_test, preds, target_names=["Legit", "Fraud"])
        print(f"\n📋 Classification Report:\n{report}")
        return float(1.0 - acc), len(X_test), {"accuracy": float(acc)}

# -------------------------------
# Run Client
# -------------------------------
fl.client.start_client(server_address="localhost:8080", client=UnifiedClient().to_client())