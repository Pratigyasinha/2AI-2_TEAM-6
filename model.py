# ==========================================
# TITANIC SURVIVAL PREDICTION PROJECT
# Logistic Regression Model
# ==========================================

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
train = pd.read_csv("data/Titanic_train.csv")
test = pd.read_csv("data/Titanic_test.csv")

print("First 5 rows of dataset:")
print(train.head())

print("\nMissing values:")
print(train.isnull().sum())

# Step 3: Data preprocessing
# Remove unnecessary columns
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# Convert categorical columns into numbers
le = LabelEncoder()
 
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])

# Step 4: Separate input and output
X = train.drop('Survived', axis=1)
y = train['Survived']

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 6: Create and train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Predict on test dataset
test_predictions = model.predict(test)

print("\nFirst 10 test predictions:")
print(test_predictions[:10])

# Step 10: Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Step 11: Save predictions to CSV
output = pd.DataFrame({
    "Predicted_Survival": test_predictions
})

output.to_csv("outputs/titanic_predictions.csv", index=False)

print("\nPredictions saved successfully in outputs folder.")