import pandas as pd

# Load datasets
train = pd.read_csv("Titanic_train.csv")
test = pd.read_csv("Titanic_test.csv")

# -------------------------------
# 1. View first few rows
# -------------------------------
print("First 5 rows of Training Data:")
print(train.head())

print("\nFirst 5 rows of Testing Data:")
print(test.head())

# -------------------------------
# 2. Dataset shape
# -------------------------------
print("\nTraining Data Shape:", train.shape)
print("Testing Data Shape:", test.shape)

# -------------------------------
# 3. Column names
# -------------------------------
print("\nColumns in dataset:")
print(train.columns)

# -------------------------------
# 4. Data types and info
# -------------------------------
print("\nTraining Data Info:")
print(train.info())

# -------------------------------
# 5. Statistical summary
# -------------------------------
print("\nStatistical Summary:")
print(train.describe())

# -------------------------------
# 6. Check missing values
# -------------------------------
print("\nMissing Values in Training Data:")
print(train.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test.isnull().sum())

# -------------------------------
# 7. Unique values (categorical understanding)
# -------------------------------
print("\nUnique values in 'Sex':", train['Sex'].unique())
print("Unique values in 'Embarked':", train['Embarked'].unique())

# -------------------------------
# 8. Value counts (distribution)
# -------------------------------
print("\nSurvival Count:")
print(train['Survived'].value_counts())

print("\nPassenger Class Distribution:")
print(train['Pclass'].value_counts())

# -------------------------------
# 9. Correlation (numeric features)
# -------------------------------
print("\nCorrelation Matrix:")
print(train.corr(numeric_only=True))