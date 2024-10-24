import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the depression dataset
depression_data_file = 'C:/Users/aivla/Desktop/Andra/College/Anul III/Semestrul I/Metode inteligente de rezolvare a problemelor reale/Laboratoare/FirstDemo/data/depression_data.csv'
depression_data = pd.read_csv(depression_data_file)

# Data Overview
print(depression_data.describe())
print(depression_data.info())

# Data Visualization
# Distribution of 'phq_score'
plt.figure(figsize=(8, 6))
sns.histplot(depression_data['phq_score'], kde=True, bins=10, color="green")
plt.title('Distribution of PHQ Score')
plt.xlabel('PHQ Score')
plt.ylabel('Frequency')
plt.show()

# PHQ Score vs Depression Diagnosis
plt.figure(figsize=(8, 6))
sns.boxplot(x='depression_diagnosis', y='phq_score', data=depression_data)
plt.title('PHQ Score by Depression Diagnosis')
plt.xlabel('Depression Diagnosis')
plt.ylabel('PHQ Score')
plt.show()

# Categorical feature analysis
sns.countplot(x='gender', hue='depression_diagnosis', data=depression_data)
plt.title('Depression Diagnosis by Gender')
plt.show()

# Outlier detection
plt.figure(figsize=(8, 6))
sns.boxplot(depression_data['phq_score'])
plt.title('Outlier Detection in PHQ Scores')
plt.show()

# Feature engineering: create binary feature for severity level
depression_data['high_severity'] = depression_data['depression_severity'].apply(lambda x: 1 if x in ['Moderately severe', 'Severe'] else 0)

print(depression_data.head())


