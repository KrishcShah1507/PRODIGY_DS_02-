# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (upload train.csv in Colab)
from google.colab import files
uploaded = files.upload()

# Read the dataset
titanic = pd.read_csv('train.csv')

# Display the first few rows
print("Dataset Head:")
print(titanic.head())

# Basic Info
print("\nDataset Info:")
print(titanic.info())

# Missing Values
print("\nMissing Values:")
print(titanic.isnull().sum())

# Handle Missing Data
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)  # Replace missing Age with median
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)  # Replace missing Embarked with mode

# Drop unnecessary columns
titanic.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Summary Statistics
print("\nSummary Statistics:")
print(titanic.describe())

# Check for duplicates
print("\nDuplicates in Dataset:")
print(titanic.duplicated().sum())

# Exploratory Data Analysis
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic, x='Survived', palette='Set2')
plt.title('Survival Counts')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(8, 6))
sns.barplot(data=titanic, x='Sex', y='Survived', palette='viridis')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(titanic['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# Survival by Pclass
plt.figure(figsize=(8, 6))
sns.barplot(data=titanic, x='Pclass', y='Survived', palette='rocket')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


