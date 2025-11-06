import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("chronic_kidney_disease.csv")

# Clean column names (remove quotes and spaces)
df.columns = df.columns.str.replace("'", "").str.strip()

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Convert numeric columns to numeric types
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Convert all text to lowercase for consistency
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Handle categorical mappings
mapping_dict = {
    'rbc': {'normal': 1, 'abnormal': 0},
    'pc': {'normal': 1, 'abnormal': 0},
    'pcc': {'present': 1, 'notpresent': 0},
    'ba': {'present': 1, 'notpresent': 0},
    'htn': {'yes': 1, 'no': 0},
    'dm': {'yes': 1, 'no': 0},
    'cad': {'yes': 1, 'no': 0},
    'appet': {'good': 1, 'poor': 0},
    'pe': {'yes': 1, 'no': 0},
    'ane': {'yes': 1, 'no': 0},
    'class': {'ckd': 1, 'notckd': 0}
}

for col, mapping in mapping_dict.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Fill missing categorical values with mode
df.fillna(df.mode().iloc[0], inplace=True)

# Define features and target
features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
target = 'class'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model training complete!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
with open("ckd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved successfully as ckd_model.pkl")
