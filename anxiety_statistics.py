import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the anxiety dataset
anxiety_data_file = 'C:/Users/aivla/Desktop/Andra/College/Anul III/Semestrul I/Metode inteligente de rezolvare a problemelor reale/Laboratoare/FirstDemo/data/anxiety_data.csv'
anxiety_data = pd.read_csv(anxiety_data_file)

# Data Overview
print(anxiety_data.describe())
print(anxiety_data.info())

# Data Visualization
# Distribution of 'gad_score'
plt.figure(figsize=(8, 6))
sns.histplot(anxiety_data['gad_score'], kde=True, bins=10)
plt.title('Distribution of GAD Score')
plt.xlabel('GAD Score')
plt.ylabel('Frequency')
plt.show()

# GAD Score vs Anxiety Diagnosis
plt.figure(figsize=(8, 6))
sns.boxplot(x='anxiety_diagnosis', y='gad_score', data=anxiety_data)
plt.title('GAD Score by Anxiety Diagnosis')
plt.xlabel('Anxiety Diagnosis')
plt.ylabel('GAD Score')
plt.show()

# Categorical feature analysis
sns.countplot(x='gender', hue='anxiety_diagnosis', data=anxiety_data)
plt.title('Anxiety Diagnosis by Gender')
plt.show()

# Outlier detection
plt.figure(figsize=(8, 6))
sns.boxplot(anxiety_data['gad_score'])
plt.title('Outlier Detection in GAD Scores')
plt.show()

# Feature engineering: create binary feature for severity level
anxiety_data['high_severity'] = anxiety_data['anxiety_severity'].apply(lambda x: 1 if x in ['Moderate', 'Severe'] else 0)

print(anxiety_data.head())
