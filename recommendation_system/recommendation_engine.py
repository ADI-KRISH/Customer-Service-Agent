""" This is a sample code for checking if the model works or not """
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("recommender_model.pkl")
fb = joblib.load("feature_builder.pkl")

df = pd.read_csv(r"path to myntra_products_catalog.csv")
product_vectors = fb.transform_products(df)

def recommend(user_context: dict, top_k=5):
    """
    user_context = {
        "budget": 2000,
        "color": "Black",
        "gender": "Men"
    }
    """

    filtered = df.copy()

    if "budget" in user_context:
        filtered = filtered[filtered["Price (INR)"] <= user_context["budget"]]

    if "color" in user_context:
        filtered = filtered[filtered["PrimaryColor"] == user_context["color"]]

    idx = filtered.index.tolist()
    X = product_vectors[idx]

    scores = model.predict_proba(X)[:, 1]

    filtered = filtered.assign(score=scores)
    filtered = filtered.sort_values("score", ascending=False)

    return filtered[[
        "ProductID",
        "ProductName",
        "Price (INR)",
        "PrimaryColor",
        "score"
    ]].head(top_k)
