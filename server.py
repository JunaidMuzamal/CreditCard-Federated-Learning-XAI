import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import EvaluateRes
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import xgboost as xgb
import joblib
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from utils import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import shap
import pandas as pd

os.makedirs("plots", exist_ok=True)
global_losses = []
final_parameters = None
model_type = "xgb"

# ----------------- STRATEGY -----------------
class SimpleStrategy(FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        print(f"\n📊 Aggregating evaluation results for round {rnd}...")
        round_losses = []
        round_accuracies = []

        for client, res in results:
            if isinstance(res, EvaluateRes):
                if res.loss is not None:
                    round_losses.append(res.loss)
                    print(f"📉 Client {client.cid} loss: {res.loss:.4f}")
                if "accuracy" in res.metrics:
                    acc = res.metrics["accuracy"]
                    round_accuracies.append(acc)
                    print(f"✅ Client {client.cid} accuracy: {acc:.4f}")

        if round_losses:
            avg_loss = sum(round_losses) / len(round_losses)
            global_losses.append(avg_loss)
            print(f"\n📈 Aggregated Loss (Round {rnd}): {avg_loss:.4f}")
        return super().aggregate_evaluate(rnd, results, failures)

    def aggregate_fit(self, rnd, results, failures):
        global final_parameters
        aggregated = super().aggregate_fit(rnd, results, failures)
        final_parameters = aggregated[0]
        return aggregated

# ----------------- MODEL SAVE -----------------
def save_global_model(model_type):
    client_path = f"model_{model_type}_client0.{ 'bst' if model_type == 'xgb' else 'pkl' }"

    if not os.path.exists(client_path):
        print(f"❌ Base model not found: {client_path}")
        return

    if model_type == "xgb":
        model = xgb.Booster()
        model.load_model(client_path)
        model.save_model("global_model.bst")
    else:
        model = joblib.load(client_path)
        joblib.dump(model, "global_model.pkl")
    print(f"💾 Global model saved as global_model.{ 'bst' if model_type == 'xgb' else 'pkl' } ✅")

# ----------------- LOSS PLOT -----------------
def plot_global_loss():
    if global_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(global_losses, marker='o')
        plt.title("Aggregated Loss Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/aggregated_loss.png")
        plt.close()
        print("📊 Saved: plots/aggregated_loss.png")

# ----------------- FINAL EVALUATION -----------------
def evaluate_final_global_model(model_type):
    print("🔍 Evaluating final global model on test set...")
    _, X_test, _, y_test = load_and_preprocess_data()

    if model_type == "xgb":
        if not os.path.exists("global_model.bst"):
            print("❌ XGBoost global model not found.")
            return
        model = xgb.Booster()
        model.load_model("global_model.bst")
        dtest = xgb.DMatrix(X_test)
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5).astype(int)
    else:
        if not os.path.exists("global_model.pkl"):
            print(f"❌ {model_type.upper()} global model not found.")
            return
        model = joblib.load("global_model.pkl")
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

    # 📋 Classification Report
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    print("\n📋 Final Classification Report:\n", report)
    with open("plots/final_classification_report.txt", "w") as f:
        f.write(report)

    # 🔷 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix (Global Model - {model_type.upper()})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("plots/final_confusion_matrix_global.png")
    plt.close()

    # 🔷 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (Global Model - {model_type.upper()})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/final_roc_curve_global.png")
    plt.close()

    # 🔷 SHAP Summary (Final FIXED Version)
    print("📈 Computing SHAP values...")
    try:
        import pandas as pd
        feature_names = [f"F{i}" for i in range(X_test.shape[1])]
        X_df = pd.DataFrame(X_test[:100], columns=feature_names)
        background = shap.sample(X_df, 50, random_state=42)

        if model_type == "xgb":
            explainer = shap.Explainer(model)
            shap_values = explainer(X_df)
            shap.summary_plot(shap_values, X_df, show=False)

        else:
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_df)

            # 🔍 Explicitly handle both formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_used = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_used = shap_values[:, :, 1]
            else:
                shap_used = shap_values

            print(f"🔎 Final SHAP used shape: {shap_used.shape}")
            shap.summary_plot(shap_used, X_df, show=False)

        plt.tight_layout()
        plt.savefig("plots/shap_summary_global.png")
        plt.close()
        print("📌 SHAP saved to: plots/shap_summary_global.png")

    except Exception as e:
        print(f"⚠️ SHAP failed for {model_type.upper()}: {e}")


# ----------------- MAIN -----------------
def main():
    global model_type
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgb", choices=["xgb", "catboost", "lr"])
    args = parser.parse_args()
    model_type = args.model_type

    print(f"🚀 Starting Flower server with model: {model_type.upper()}...\n")
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=SimpleStrategy(),
        )
    finally:
        plot_global_loss()
        if final_parameters:
            save_global_model(model_type)
            evaluate_final_global_model(model_type)

if __name__ == "__main__":
    main()