# =============================
# 1. Import Libraries
# =============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# =============================
# 2. Load Dataset
# =============================
file_path = r"C:\Users\satya\tata\Satyam.xlsx"

df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nColumns:\n", df.columns)

# =============================
# 3. Data Preprocessing
# =============================

# Payment mapping (safe conversion)
payment_map = {
    "On-time": 0,
    "Late": 1,
    "Missed": 2
}

for col in df.columns:
    if "Month" in col:
        df[col] = df[col].map(payment_map).fillna(df[col])

# Missing value handling
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# =============================
# 4. Feature & Target Selection|
# =============================

features = [
    "Income",
    "Credit_Score",
    "Credit_Utilization",
    "Missed_Payments",
    "Debt_to_Income_Ratio"
]

target = "Delinquent_Account"

# Check target column
if target not in df.columns:
    raise Exception(f"❌ Target column '{target}' not found in dataset")

# Keep only valid features
features = [col for col in features if col in df.columns]

if len(features) == 0:
    raise Exception("❌ No valid feature columns found")

X = df[features]
y = df[target]

# Ensure numeric only
X = X.select_dtypes(include=["number"])

# =============================
# 5. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. Train Model
# =============================
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# =============================
# 7. Predictions
# =============================
y_pred = model.predict(X_test)

# =============================
# 8. Evaluation
# =============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============================
# 9. Feature Importance
# =============================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)
