import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_excel("Satyam.xlsx")

# 2. Select features & target
X = df[["Income", "Credit_Score", "Credit_Utilization"]]
y = df["Delinquent_Account"]

# 3. Fill missing values
X = X.fillna(X.mean())

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)

# 7. Accuracy
print("Accuracy:", model.score(X_test, y_test))