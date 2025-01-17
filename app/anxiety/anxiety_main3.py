import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import re
from nltk.corpus import stopwords


def preprocess_text(text):
    # Preprocessing text by removing non-alphabetical characters
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().split()
    return [word for word in tokens if word not in stop_words]


def get_bert_embeddings(texts, tokenizer, model):
    # Tokenize and get BERT embeddings
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling for sentence embeddings


def main():
    # Load data
    data = pd.read_csv("../new_data/anxiety_data.csv")
    data['Processed_Text'] = data['Text'].apply(preprocess_text)

    # Initialize BERT tokenizer and model from Hugging Face
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Get embeddings for each text
    texts = [' '.join(text) for text in data['Processed_Text']]
    X = get_bert_embeddings(texts, tokenizer, model)

    # Convert target variable into binary format (1 for anxiety, 0 for normal)
    y = data['Target'].apply(lambda x: 1 if x == "anxiety" else 0)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # GridSearch with limited param_grid for efficiency
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 0.01],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    # Evaluate the best model
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print("Validation Accuracy:", accuracy)
    print("Best Parameters:", grid.best_params_)


if __name__ == '__main__':
    main()
