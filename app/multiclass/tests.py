import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
test_list = [
        "I feel sleepy all the time.",
        "Can you give me the source of the comment?",
        "Answer me!",
        "Much love and respect. Happy birthday",
        "thats a you problem girl",
        "I want to kill myself...",
        "So many happy people all around me, hate it",
        "Wanna smoke outside?",
        "Do you also have mood swings?",
        "I have a hard time when I think about the tomorrow test",
        "I want to kill myself",
    "depressed",
    "sweating all over thinking about tomorrow",
    "anxious",
    "I feel usure",
    "love and kind regards",
    "I didn't felt anything for his accident. Nothing at alL."
    ]

def neural_network_test():
    # Load the Keras model
    model = load_model('../persistency/multiclass_classifiers/model_3_class_classifier.h5')

   

    # Load the same transformer model used for training
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert the test sentences into embeddings
    test_embeddings = transformer_model.encode(test_list)

    # Make predictions
    predictions = model.predict(test_embeddings)  # Get probabilities
    predicted_classes = np.argmax(predictions, axis=1)  # Get class indices

    # Map numeric predictions back to labels
    labels = ["normal", "anxiety", "depression"]
    results = [labels[pred] for pred in predicted_classes]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")
def random_forests():


    # Load the pretrained classifier
    #classifier_t = joblib.load('../persistency/multiclass_classifiers/model_3_class_classifier_random_forests_best_new_dataset_trained.joblib')

    classifier_t = joblib.load(
        '../persistency/multiclass_classifiers/tests_on_diff_datasets/model_3_class_classifier_random_forests_best_new_dataset_trained2.joblib')

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')

    # Fit the vectorizer on the test data
    #tfidf_vectorizer.fit(test_list)

    # Transform the test list
    #test_embeddings = tfidf_vectorizer.transform(test_list).toarray()
    example_sentences = preprocess_text(pd.Series(test_list))

    # Make predictions
    predictions = classifier_t.predict(example_sentences)

    # Map numeric predictions back to labels
    labels = ["anxiety", "depression", "normal"]
    results = [labels[pred] for pred in predictions]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")


def random_forests_new():
    
    # Load the pretrained classifier
    classifier_t = joblib.load('../persistency/multiclass_classifiers/optimized_model.joblib')

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')

    # Fit the vectorizer on the test data
    #tfidf_vectorizer.fit(test_list)

    # Transform the test list
    #test_embeddings = tfidf_vectorizer.transform(test_list).toarray()

    # Make predictions
    predictions = classifier_t.predict(test_list)

    # Map numeric predictions back to labels
    labels = ["normal", "anxiety", "depression"]
    results = [labels[pred] for pred in predictions]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")

def preprocess_text(texts):
    """Apply consistent preprocessing to the input texts."""
    return texts.str.lower()
def mlp_classifier_new():

    # Load the pretrained classifier
    classifier_t = joblib.load('../persistency/multiclass_classifiers/tests_on_diff_datasets/model_3_class_classifier_mlp_classifier_on_balanced.joblib')

    # Load the same transformer model used for training
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert the test sentences into embeddings
    test_embeddings = transformer_model.encode(test_list)

    # Make predictions
    predictions = classifier_t.predict(test_embeddings)

    # Map numeric predictions back to labels
    labels = ["normal", "anxiety", "depression"]
    results = [labels[pred] for pred in predictions]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")



if __name__ == '__main__':
    random_forests()
    #random_forests_new()
    #mlp_classifier_new()
