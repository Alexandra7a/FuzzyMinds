import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


def plot_training_history(history):
    """Plot the training and validation accuracy and loss over epochs."""
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy vs Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss vs Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def preprocess_text(texts):
    """Apply consistent preprocessing to the input texts."""
    return texts.str.lower()

def make_confusion_matrix(y_test, y_pred, class_labels):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.show()

def make_model():
    # Load dataset
    #data = pd.read_csv("../data/test_dataset_multi.csv")
    data=pd.read_csv("C:/Users/Alexandra/Documents/GitHub/projects-fuzzyminds/BalancedDatasetShuffled (6).csv")
    data = data.dropna(subset=['Text'])

    # Encode labels using LabelEncoder for robustness
    label_encoder = LabelEncoder()
    data['Target'] = label_encoder.fit_transform(data['Target'])
    class_labels = label_encoder.classes_



    joblib.dump(label_encoder, "../persistency/multiclass_classifiers/tests_on_diff_datasets/label_encoder_random_forests.joblib")
    print(label_encoder)
    print(label_encoder.classes_)
    pass
    # Preprocess text
    data['Text'] = preprocess_text(data['Text'])

    # Split the data into training and testing sets
    X = data['Text']
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a pipeline with TfidfVectorizer and RandomForestClassifier
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english'),
                          RandomForestClassifier(n_estimators=200, random_state=42))

    # Evaluate using cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

    # Train the model
    model.fit(X_train, y_train)


    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, "../persistency/multiclass_classifiers/tests_on_diff_datasets/model_3_class_classifier_random_forests_best_new_dataset_trained3.joblib")

    # Example usage
    example_sentences = [
        "I'm having trouble sleeping because I'm constantly worried.",
        "I feel happy and content with my life.",
        "Sometimes I just want to cry without reason."
    ]

    # Preprocess and predict
    example_sentences = preprocess_text(pd.Series(example_sentences))


    predictions = model.predict(example_sentences)
    for sentence, label in zip(example_sentences, predictions):
        label_name = class_labels[label]
        print(f"'{sentence}' is classified as: {label_name}")

    make_confusion_matrix(y_test, y_pred, class_labels)

def main():
    start_time = time.time()
    make_model()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()