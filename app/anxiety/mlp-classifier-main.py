from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['No Anxiety', 'Anxiety'], yticklabels=['No Anxiety', 'Anxiety'])
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.show()


def make_model():
    data = pd.read_csv("../new_data/anxiety_data.csv")
    data['Target'] = data['Target'].apply(lambda x: 1 if x == "anxiety" else 0)

    y = data['Target']
    X = data['Text']

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(X.tolist())

    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, target_names=['No Anxiety', 'Anxiety'], output_dict=True)
    print("\nPrecision and Recall for Each Class:")
    print(
        f"No Anxiety: Precision = {report['No Anxiety']['precision']:.2f}, Recall = {report['No Anxiety']['recall']:.2f}")
    print(f"Anxiety: Precision = {report['Anxiety']['precision']:.2f}, Recall = {report['Anxiety']['recall']:.2f}")

    print(f"\nAverage Precision: {report['weighted avg']['precision']:.2f}")

    make_confusion_matrix(y_test, y_pred)

    joblib.dump(classifier, '../persistency/classifiers/all-MiniLM-L6-v2_anxiety_classifier.joblib')

def testing(classifier_t,transformer_model):
    test_list = [
        "I feel sleepy all the time.",
        "Can you give me the source of the comment?",
        "Answer me!",
        "Much love and respect. Happy birthday",
        "thats a you problem girl",
        "I sometimes feel like giving up...",
        "So many happy people all around me, hate it",
        "Wanna smoke outside?",
        "Do you also have mood swings?",
        "I have a hard time when I think about the tomorrow test"
    ]

    # Convert the test sentences into embeddings
    test_embeddings = transformer_model.encode(test_list)

    # Make predictions
    predictions = classifier_t.predict(test_embeddings)

    # Map numeric predictions back to labels
    labels = ["No Anxiety", "Anxiety"]
    results = [labels[pred] for pred in predictions]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")

def main():#make_model()
    model=SentenceTransformer('all-MiniLM-L6-v2')
    classifier=joblib.load('../persistency/classifiers/all-MiniLM-L6-v2_anxiety_classifier.joblib')
    testing(classifier, model)
    

if __name__ == '__main__':
    main()
