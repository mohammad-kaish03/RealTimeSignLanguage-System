import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from config import DATA_PATH, MODEL_PATH

# Check dataset first
if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found. Run data_collection.py first.")
    exit()

# Ensure model folder exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"✅ Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
