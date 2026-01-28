# ============================================================================
# BEST MODEL: SVC (Support Vector Classifier) with Hyperparameter Tuning
# ============================================================================

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================================
# TASK-1: DATA LOADING
# ============================================================================


# Load the dataset
file_path = "Social_Network_Ads.csv"
df = pd.read_csv(file_path)


# Drop User ID as it's not required
df = df.drop('User ID', axis=1)

df_engineered = df.copy()

# Age binning
df_engineered["Age_bin"] = pd.cut(
    df_engineered["Age"],
    bins=[0, 25, 35, 45, 100],
    labels=["Young", "Middle", "Mature", "Old"]
)
df_engineered = df_engineered.drop("Age", axis=1)

# ---- 2.3 Define Column Types ----
numerical_cols = ["EstimatedSalary"]
categorical_cols = ["Gender", "Age_bin"]
binary_categorical_cols = ["Gender"]
target_col = "Purchased"

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


# Prepare features and target
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df[target_col]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Define feature types
numerical_features = ['EstimatedSalary']
onehot_features = ['Gender', 'Age_bin']

# Create the ColumnTransformer for numerical scaling and categorical encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features)
    ],
    remainder='passthrough'
)

# Construct the full preprocessing pipeline
full_preprocessing_pipeline = Pipeline(steps=[
    ('age_binner', AgeBinner()),
    ('preprocessor', preprocessor)
])

# Best hyperparameters from GridSearchCV: C=10, gamma='auto'
best_model = Pipeline(steps=[
    ('preprocessor', full_preprocessing_pipeline),
    ('classifier', SVC(C=10, gamma='auto', probability=True, random_state=42))
])

best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased']))

with open("best_model_Social_ads.pkl", "wb") as f:
    pickle.dump(best_model, f)

# print pickle confirmation
print("Pickle file created successfully as :", "best_model_Social_ads.pkl")
