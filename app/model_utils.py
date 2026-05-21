"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import pickle
import pandas as pd

model = pickle.load(open("data/model.pkl", "rb"))
preprocessor = pickle.load(open("data/col_transformer.pkl", "rb"))


def preprocess(features: list[float]) -> list[float]:
    """
    Takes raw features and applies necessary preprocessing (e.g. scaling).
    """
    input_df = pd.DataFrame([features], columns=preprocessor.feature_names_in_)
    transformed = preprocessor.transform(input_df)
    return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    processed_features = preprocess(features)

    # model.predict() expects a 2D array-like; preprocess already returns a 2D DataFrame.
    prediction = model.predict(processed_features)
    return int(prediction[0])


# if __name__ == "__main__":
#     sample = [600, "France", "Female", 40, 3, 60000, 2, 1, 1, 50000]
#     print(f"Input:      {sample}")
#     print(f"Prediction: {predict_churn(sample)}")
