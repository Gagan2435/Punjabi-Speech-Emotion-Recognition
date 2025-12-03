import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1️⃣ Load cleaned features
df = pd.read_csv("feature_ext_output_cleaned.csv")

# 2️⃣ Prepare X and y
X = df.drop(columns=['participant', 'emotion', 'filename', 'full_path', 'relative_path'])
y = df['emotion']

# Confirm unique labels
print("\n✅ Unique labels before encoding:", y.unique())

# 3️⃣ Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n✅ Label mapping:")
for idx, cls in enumerate(label_encoder.classes_):
    print(f"{idx}: {cls}")

# 4️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6️⃣ Train Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight='balanced_subsample',
    max_depth=15
)

model.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 8️⃣ Save artifacts
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✅ Model, scaler, and label encoder saved successfully and ready for testing.")
