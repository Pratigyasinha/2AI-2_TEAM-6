# ==========================================
# MODEL EVALUATION - TITANIC PROJECT
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load cleaned dataset
train = pd.read_csv("data/clean_train.csv")

# Step 2: Separate input and output
X = train.drop('Survived', axis=1)
y = train['Survived']

# Step 3: Train-test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train model (needed for evaluation)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# ==========================================
# EVALUATION STARTS HERE
# ==========================================

print("===== MODEL EVALUATION =====")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Survived', 'Survived'],
    yticklabels=['Not Survived', 'Survived']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save graph
plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved in outputs folder.")