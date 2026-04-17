# ==========================================
# TITANIC DATA PREPROCESSING
# ==========================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
train = pd.read_csv("data/Titanic_train.csv")
test = pd.read_csv("data/Titanic_test.csv")

print("Before Preprocessing:")
print(train.isnull().sum())

# Step 2: Drop unnecessary columns
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 3: Handle missing values

# Fill Age with mean
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

# Fill Embarked with most frequent value
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# Fill Fare (only in test sometimes)
if 'Fare' in test.columns:
    test['Fare'].fillna(test['Fare'].mean(), inplace=True)

# Step 4: Convert categorical data into numbers
le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])

# Step 5: Check after preprocessing
print("\nAfter Preprocessing:")
print(train.isnull().sum())

# Step 6: Save cleaned data (optional but recommended)
train.to_csv("data/clean_train.csv", index=False)
test.to_csv("data/clean_test.csv", index=False)

print("\nPreprocessing completed and cleaned data saved!")