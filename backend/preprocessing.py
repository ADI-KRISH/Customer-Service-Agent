import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer

class FeatureBuilder:
    def __init__(self):
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.brand_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.color_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.gender_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = MinMaxScaler()

    def fit(self, df: pd.DataFrame):
        self.brand_encoder.fit(df[["ProductBrand"]])
        self.color_encoder.fit(df[["PrimaryColor"]])
        self.gender_encoder.fit(df[["Gender"]])
        self.scaler.fit(df[["Price (INR)"]])

    def transform_products(self, df: pd.DataFrame):
        text_embeddings = self.text_model.encode(
            (df["ProductName"] + " " + df["Description"]).tolist(),
            show_progress_bar=True
        )

        brand_feat = self.brand_encoder.transform(df[["ProductBrand"]])
        color_feat = self.color_encoder.transform(df[["PrimaryColor"]])
        gender_feat = self.gender_encoder.transform(df[["Gender"]])
        numeric_feat = self.scaler.transform(df[["Price (INR)"]])

        return np.hstack([
            text_embeddings,
            brand_feat,
            color_feat,
            gender_feat,
            numeric_feat
        ])
