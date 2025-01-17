import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from gensim.models import Word2Vec
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# nltk.download('stopwords')


def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().split()
    return [word for word in tokens if word not in stop_words]


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    for word in words:
        if word in vocabulary:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',  # Set color map to blue
                xticklabels=['Anxiety', 'No Anxiety'], yticklabels=['Anxiety', 'No Anxiety'])
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.show()


def main():
    # Load data

    data = pd.read_csv("../new_data/anxiety_data.csv")
    data = data.iloc[:500]  # Small data subset

    data['Processed_Text'] = data['Text'].apply(preprocess_text)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=data['Processed_Text'], vector_size=200, window=10, min_count=2, workers=4)
    vocabulary = set(word2vec_model.wv.index_to_key)

    # Generate feature vectors for each text
    X = np.array([average_word_vectors(words, word2vec_model.wv, vocabulary, 200) for words in data['Processed_Text']])
    y = data['Target'].apply(lambda x: 1 if x == "anxiety" else 0)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM and GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 0.01],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    # Evaluation
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print("Validation Accuracy:", accuracy)
    print("Validation Precision:", precision)
    print("Validation Recall:", recall)
    print("Best Parameters:", grid.best_params_)

    # Make confusion matrix
    make_confusion_matrix(y_val, y_pred)


if __name__ == '_main_':
    main()
