from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.show()

def main():
    data = pd.read_csv("../new_data/depression_data.csv")
    data['Target'] = data['Target'].apply(lambda x: 1 if x == "depression" else 0)

    y = data['Target']
    X = data['Text']

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(X.tolist())

    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, target_names=['No Depression', 'Depression'], output_dict=True)
    print("\nPrecision and Recall for Each Class:")
    print(f"No Depression: Precision = {report['No Depression']['precision']:.2f}, Recall = {report['No Depression']['recall']:.2f}")
    print(f"Depression: Precision = {report['Depression']['precision']:.2f}, Recall = {report['Depression']['recall']:.2f}")

    print(f"\nAverage Precision: {report['weighted avg']['precision']:.2f}")

    make_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    main()
