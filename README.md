
# Federated Learning for Credit Card Fraud Detection

This project implements a **federated learning framework** using [Flower](https://flower.dev) to collaboratively train models for credit card fraud detection without centralizing the data. It supports three model types: **XGBoost**, **CatBoost**, and **Logistic Regression**, and includes SHAP-based explainability and performance evaluations.

---

## 📁 Project Structure

```
.
├── client.py                  # Federated client script
├── server.py                  # Federated server script
├── run_federated_system.py   # Launcher for server and multiple clients
├── utils.py                  # Data loading, preprocessing, training, plotting
├── creditcard.csv            # Dataset (Kaggle Credit Card Fraud)
├── plots/                    # Generated plots (loss, ROC, SHAP, confusion matrix)
├── model_*.pkl               # Saved per-client models
├── global_model.pkl/.bst     # Saved global model (based on selected model_type)
```

---

## 🚀 Features

- ✅ Federated Learning using Flower with 5 simulated clients.
- ✅ Supports 3 model types: `xgb`, `catboost`, and `lr`.
- ✅ Applies SMOTE to balance data before client training.
- ✅ Saves per-round aggregated loss plot.
- ✅ Evaluates final global model: classification report, confusion matrix, ROC.
- ✅ SHAP explainability for global model (with debug logs).
- ✅ Separate plots and model files per client.

---

## 🧪 Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Preprocessing:
  - Standard scaling
  - SMOTE oversampling on the training data
  - 80-20 train-test split (with stratification)

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

> Dependencies include: `flwr`, `xgboost`, `catboost`, `shap`, `matplotlib`, `scikit-learn`, `imblearn`, `seaborn`, `pandas`, `joblib`

---

## 🔧 How to Run

### Step 1: Start Federated Learning

Use the unified launcher:

```bash
python run_federated_system.py --model_type catboost
```

Available options:
- `--model_type xgb` → XGBoost
- `--model_type catboost` → CatBoost
- `--model_type lr` → Logistic Regression

This will:
- Launch the server in a new PowerShell window
- Launch 5 federated clients in separate PowerShell windows

> Note: Ensure you're on Windows and have `PowerShell` available. For Linux/macOS, modify the subprocess calls accordingly.

---

## 📊 Outputs

After training completes:

### 🔹 `plots/aggregated_loss.png`  
Shows the average loss across rounds.

### 🔹 `plots/final_confusion_matrix_global.png`  
Confusion matrix of the global model on test set.

### 🔹 `plots/final_roc_curve_global.png`  
ROC curve of the global model.

### 🔹 `plots/shap_summary_global.png`  
SHAP value plot (summary) for top features of the global model.

### 🔹 `plots/final_classification_report.txt`  
Detailed classification report.

---

## 📌 Known Issues

- SHAP summary plots work perfectly with XGBoost. For `catboost` and `lr`, SHAP might sometimes show only one feature. This is a limitation of KernelExplainer with flat predictions and can vary based on model internals. Debug logs are enabled.
- Flower `start_server()` is deprecated. This project uses it for simplicity but can be updated to `flower-superlink`.

---

## 🧠 Future Improvements

- Add custom SHAP feature importances per client.
- Replace `KernelExplainer` with `TreeExplainer` (for CatBoost) if supported.
- Add async aggregation strategies and real-time dashboards.
- Enable model checkpointing and recovery.

---

## 🏷 License

MIT License © 2025
