import gradio as gr
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# IMPORTANT: Custom Transformer (MUST exist before pickle load)
# =====================================================
class AgeBinner(BaseEstimator, TransformerMixin):
    """Custom transformer to bin age into categories"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['Age_bin'] = pd.cut(
            X_copy['Age'],
            bins=[0, 25, 35, 45, 100],
            labels=['Young', 'Middle', 'Mature', 'Old'],
            right=True
        )
        X_copy = X_copy.drop('Age', axis=1)
        return X_copy

# =====================================================
# Load trained model
# =====================================================
with open("best_model_Social_ads.pkl", "rb") as f:
    model = pickle.load(f)

# =====================================================
# Prediction function
# =====================================================
def predict_purchase(Gender, Age, EstimatedSalary):
    input_df = pd.DataFrame(
        [[Gender, Age, EstimatedSalary]],
        columns=["Gender", "Age", "EstimatedSalary"]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        return f"✅ Purchased (Probability: {probability:.2f})"
    else:
        return f"❌ Not Purchased (Probability: {probability:.2f})"

# =====================================================
# Gradio Interface
# =====================================================
app = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Slider(18, 70, step=1, label="Age"),
        gr.Number(value=50000, label="Estimated Salary"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Social Network Ads Purchase Prediction",
)

app.launch(share=True)
