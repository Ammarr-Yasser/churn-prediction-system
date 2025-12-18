# churn-prediction-system
Telecom customer churn study with multiple classical ML models. The repo ships a lightweight Flask UI that loads pre-trained pipelines, evaluates them on held-out data, and surfaces the best model with sample predictions.

## Repository layout
- `app.py`: Flask app that loads pipelines, scores the test split, and renders metrics plus sample predictions.
- `templates/index.html`: Simple UI to display the best model and a 50-row preview of predictions.
- `all_churn_pipelines.pkl`: Serialized dictionary of scikit-learn pipelines keyed by model name.
- `X_test_churn.csv` / `y_test_churn.csv`: Held-out feature matrix and labels used for evaluation and the demo table.
- `telecom_churn_coding.ipynb`: Notebook that explores the dataset, trains pipelines, and exports the artifacts.
- `ML_Project.pdf`: Project write-up.
- `telecom_churn_full.csv`: Source dataset (Telecom Egypt / WE).

## Prerequisites
- Python 3.10+ recommended
- pip

## Setup
```bash
python -m venv .venv
# Windows
.\\.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install flask pandas scikit-learn joblib
# Optional for experimenting in the notebook
pip install jupyter
```

## Run the web demo
```bash
python app.py
# App will start at http://127.0.0.1:5000/
```
The app will:
1) Load all pipelines from `all_churn_pipelines.pkl`.
2) Score `X_test_churn.csv` vs `y_test_churn.csv`.
3) Compute Accuracy, Precision, Recall, F1, and AUC; pick the best model by AUC.
4) Show metrics for every model and the first 50 predictions (true label, predicted label, probability) for the best model.

## Models included
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine (RBF)
- Multilayer Perceptron

## Updating models or data
1) Open `telecom_churn_coding.ipynb` and retrain/adjust pipelines as needed.
2) Export updated artifacts: `all_churn_pipelines.pkl`, `X_test_churn.csv`, and `y_test_churn.csv`.
3) Restart the Flask app; the UI will reflect new metrics and predictions.

## Notes and assumptions
- Probability threshold is fixed at 0.5 for classification in the UI.
- Metrics are derived from the provided test split; swap in your own test data if desired.
- The app runs in debug mode by default; set `debug=False` or use a production WSGI server for deployment.
