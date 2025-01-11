import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    data = pd.read_csv(
        "../data/BalancedDatasetShuffled.csv")
    data = data.dropna(subset=['Text'])

    print(data['Target'].head())

    # Encode labels using LabelEncoder for robustness
    label_encoder = LabelEncoder()
    data['Target'] = label_encoder.fit_transform(data['Target'])
    class_labels = label_encoder.classes_

    print(data['Target'].head())
    # Preprocess text
    data['Text'] = preprocess_text(data['Text'])

    print(data['Target'].head())

    # Split the data into training and testing sets
    X = data['Text']
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize SentenceTransformer model
    model_sentence = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the texts into sentence embeddings
    X_train_embedded = model_sentence.encode(X_train.tolist(), convert_to_tensor=True)
    X_test_embedded = model_sentence.encode(X_test.tolist(), convert_to_tensor=True)

    # Build a simple ANN model using MLPClassifier
    ann_model = MLPClassifier(hidden_layer_sizes=(128, 80), max_iter=100, random_state=42)

    # Train the model
    ann_model.fit(X_train_embedded, y_train)

    # Evaluate the model
    y_pred = ann_model.predict(X_test_embedded)

    # Print Classification Report and Accuracy
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(ann_model, "../persistency/multiclass_classifiers/tests_on_diff_datasets/model_3_class_classifier_mlp_classifier_on_balanced.joblib")

    # Example usage
    example_sentences = [
        "I'm having trouble sleeping because I'm constantly worried.",
        "I feel happy and content with my life.",
        "Sometimes I just want to cry without reason."
    ]

    # Preprocess and predict
    example_sentences = preprocess_text(pd.Series(example_sentences))
    example_embeddings = model_sentence.encode(example_sentences.tolist(), convert_to_tensor=True)
    example_predictions = ann_model.predict(example_embeddings)

    for sentence, label in zip(example_sentences, example_predictions):
        label_name = class_labels[label]
        print(f"'{sentence}' is classified as: {label_name}")

    # Create confusion matrix
    make_confusion_matrix(y_test, y_pred, class_labels)


def main():
    start_time = time.time()
    make_model()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()