import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import os

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

CLIENT_CHUNKS = []
X_TEST, Y_TEST = None, None

def load_and_preprocess_data():
    global CLIENT_CHUNKS, X_TEST, Y_TEST

    df = pd.read_csv("creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE Class Distribution: {dict(pd.Series(y_bal).value_counts())}")

    indices = np.random.permutation(len(X_bal))
    X_bal = X_bal[indices]
    y_bal = y_bal[indices]

    CLIENT_CHUNKS = []
    total = len(X_bal)
    chunk_size = int(0.6 * total)

    for i in range(5):
        start = int(i * 0.2 * total)
        end = start + chunk_size
        CLIENT_CHUNKS.append((X_bal[start:end], y_bal[start:end]))

    X_TEST, Y_TEST = X_test, y_test
    return None, X_test, None, y_test

def load_client_data(index):
    if not CLIENT_CHUNKS:
        load_and_preprocess_data()
    return *CLIENT_CHUNKS[index], X_TEST, Y_TEST

# --- TRAINING FUNCTIONS ---

def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals_result = {}
    model = xgb.train(params, dtrain, num_boost_round=20,
                      evals=[(dtrain, "train")],
                      evals_result=evals_result, verbose_eval=True)
    logloss = evals_result["train"]["logloss"]
    plot_loss_curve(logloss, "xgboost")
    return model

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6,
                               loss_function='Logloss', verbose=0)
    model.fit(X_train, y_train)
    return model

def train_logistic(X_train, y_train):
    df = pd.DataFrame(X_train, columns=[f"F{i}" for i in range(X_train.shape[1])])
    model = LogisticRegression(max_iter=1000, solver='lbfgs',
                               class_weight='balanced', n_jobs=-1)
    model.fit(df, y_train)
    return model

# --- PREDICTION ---

def predict_model(model, X, model_type):
    if model_type == "xgboost":
        return model.predict(xgb.DMatrix(X))
    return model.predict_proba(X)[:, 1]

# --- EVALUATION ---

def evaluate_model(model, X_test, y_test, model_type="xgboost", suffix=""):
    y_proba = predict_model(model, X_test, model_type)
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📋 Classification Report ({model_type}):\n",
          classification_report(y_test, y_pred))
    print(f"📌 Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, suffix)
    plot_roc_curve(y_test, y_proba, suffix)
    plot_pr_curve(y_test, y_proba, suffix)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# --- PLOTTING HELPERS ---

def plot_loss_curve(logloss, model_name):
    plt.figure(figsize=(8, 4))
    plt.plot(logloss, marker="o")
    plt.title(f"{model_name.upper()} Training Log Loss")
    plt.xlabel("Boosting Round")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/logloss_{model_name}.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix{suffix}.png")
    plt.close()

def plot_roc_curve(y_true, y_proba, suffix=""):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"plots/roc_curve{suffix}.png")
    plt.close()

def plot_pr_curve(y_true, y_proba, suffix=""):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/pr_curve{suffix}.png")
    plt.close()