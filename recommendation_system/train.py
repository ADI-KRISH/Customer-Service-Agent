import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import FeatureBuilder

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:/Users/GS Adithya Krishna/Desktop/internship/data/myntra_products_catalog.csv")

# -----------------------------
# Feature engineering
# -----------------------------
fb = FeatureBuilder()
fb.fit(df)
X = fb.transform_products(df)

# -----------------------------
# Build proxy relevance labels (NO LEAKAGE)
# Use text similarity ONLY for labeling
# -----------------------------
text_embeddings = fb.text_model.encode(
    (df["ProductName"] + " " + df["Description"]).tolist(),
    show_progress_bar=False
)

# Choose a reference prototype (e.g., average embedding)
prototype = text_embeddings.mean(axis=0).reshape(1, -1)

similarity_scores = cosine_similarity(text_embeddings, prototype).flatten()

# Top 30% most semantically central products = relevant
threshold = np.percentile(similarity_scores, 70)
y = (similarity_scores >= threshold).astype(int)

print("Positive samples:", y.sum(), "/", len(y))

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train LightGBM model
# -----------------------------
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation (classification sanity check)
# -----------------------------
y_pred = model.predict(X_test)

print("\n=== Classification Metrics (Sanity Check) ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

# -----------------------------
# Ranking Metrics (REAL evaluation)
# -----------------------------
def precision_at_k(model, X, y_true, k=5):
    scores = model.predict_proba(X)[:, 1]
    top_k = np.argsort(scores)[-k:][::-1]
    return y_true[top_k].sum() / k

def recall_at_k(model, X, y_true, k=5):
    scores = model.predict_proba(X)[:, 1]
    top_k = np.argsort(scores)[-k:][::-1]
    return y_true[top_k].sum() / y_true.sum()

print("\n=== Recommender Metrics ===")
print("Precision@5:", precision_at_k(model, X_test, y_test))
print("Recall@5   :", recall_at_k(model, X_test, y_test))

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, "recommender_model.pkl")
joblib.dump(fb, "feature_builder.pkl")

print("\nLightGBM model trained (NO leakage).")
