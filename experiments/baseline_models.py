import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# -----------------------------
# Robust path handling (CRITICAL)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Logistic Regression with scaling
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000))
])

models = {
    "Logistic Regression": log_reg,
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

names, scores = [], []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    names.append(name)
    scores.append(accuracy_score(y_val, preds))

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.bar(names, scores)
plt.ylabel("Validation Accuracy")
plt.title("Baseline Model Comparison")

output_path = os.path.join(PLOTS_DIR, "baseline_comparison.png")
plt.savefig(output_path, dpi=120, bbox_inches="tight")
plt.close()

print(f"Saved plot to: {output_path}")
