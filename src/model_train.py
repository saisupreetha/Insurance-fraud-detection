# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("insurance_claims.csv")

# Drop unnecessary columns
df = df.drop(columns=['_c39', 'policy_number'], errors='ignore')

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# -------------------------
# Encode Categorical Columns
# -------------------------
label_encoders = {}

cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------
# Split Features & Target
# -------------------------
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Save feature order
joblib.dump(X.columns.tolist(), "feature_order.pkl")

# -------------------------
# Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Apply SMOTE
# -------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------------
# Train XGBoost Model
# -------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Check class order
print("Class order:", model.classes_)

# -------------------------
# Save Everything
# -------------------------
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and encoders saved successfully.")
