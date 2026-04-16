
# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 2. LOAD DATASET
# ==============================
df = pd.read_csv('train.csv')

print("\n===== DATA PREVIEW =====")
print(df.head())


# ==============================
# 3. DATA CLEANING
# ==============================

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Create FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop old columns
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# ==============================
# 5. DATA VISUALIZATION (SAVED)
# ==============================

plt.figure(figsize=(16, 12))

# Plot 1
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')

# Plot 2
plt.subplot(2, 2, 2)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')

# Plot 3
plt.subplot(2, 2, 3)
plt.hist(df['Age'], bins=20)
plt.title('Age Distribution')

# Plot 4
plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('visualizations.png')   # ✅ Saves graph
plt.show()


# ==============================
# 6. DEFINE FEATURES & TARGET
# ==============================
X = df.drop('Survived', axis=1)
y = df['Survived']


# ==============================
# 7. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 8. MODEL BUILDING
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ==============================
# 9. PREDICTION
# ==============================
y_pred = model.predict(X_test)


# ==============================
# 10. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)


# ==============================
# 11. CONFUSION MATRIX (SAVED)
# ==============================

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig('confusion_matrix.png')  # ✅ Saves graph
plt.show()


# ==============================
# 12. SAVE OUTPUT TO FILE
# ==============================

with open("output.txt", "w") as f:
    f.write("===== TITANIC MODEL OUTPUT =====\n\n")
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\n✅ Output saved to output.txt")
print("✅ Graphs saved as images")
