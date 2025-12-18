from flask import Flask, render_template
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load full pipelines and data
pipelines = joblib.load(BASE_DIR / "all_churn_pipelines.pkl")
X_test = pd.read_csv(BASE_DIR / "X_test_churn.csv")
y_test = pd.read_csv(BASE_DIR / "y_test_churn.csv").squeeze()

metrics_table = []
best_model_name = None
best_auc = -1.0

for name, clf in pipelines.items():
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    metrics_table.append({
        "model_name": name,
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "auc": round(auc, 3),
    })

    if auc > best_auc:
        best_auc = auc
        best_model_name = name

best_model = pipelines[best_model_name]

probs = best_model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

results_df = X_test.copy()
results_df["true_churn"] = y_test.values
results_df["predicted_churn"] = preds
results_df["churn_probability"] = probs

sample_rows = results_df.head(50).to_dict(orient="records")
sample_columns = results_df.columns

@app.route("/")
def index():
    return render_template(
        "index.html",
        best_model_name=best_model_name,
        metrics_table=metrics_table,
        rows=sample_rows,
        columns=sample_columns
    )

if __name__ == "__main__":
    app.run(debug=True)
