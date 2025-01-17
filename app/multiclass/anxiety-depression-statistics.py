from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# Class distribution as bar chart
def plot_class_distribution_bar(df, class_column):
    class_counts = df[class_column].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, dodge=False, palette="viridis")
    plt.title("Class Distribution (Bar Chart)")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.legend([], [], frameon=False)
    plt.show()


# Class distribution as pie chart
def plot_class_distribution_pie(df, class_column):
    class_counts = df[class_column].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("viridis", len(class_counts)))
    plt.title("Class Distribution (Pie Chart)")
    plt.show()


# Text length distribution
def plot_text_length_distribution(df, text_column, class_column):
    df['text_length'] = df[text_column].apply(len)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=class_column, y='text_length', dodge=False, palette="viridis")
    plt.title("Text Length Distribution (Box Plot)")
    plt.xlabel("Class")
    plt.ylabel("Text Length")
    plt.show()


# Function to calculate and plot TF-IDF scores specific to a class
def plot_class_specific_words(df, target_class, top_n=10):
    # Separate texts for the target class and the rest
    class_texts = df[df["Target"] == target_class]["Text"]
    other_texts = df[df["Target"] != target_class]["Text"]

    all_texts = pd.concat([class_texts, other_texts], axis=0).tolist()

    # Compute TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Split TF-IDF matrix into target class and other classes
    target_class_tfidf = tfidf_matrix[:len(class_texts)].mean(axis=0).A1
    other_class_tfidf = tfidf_matrix[len(class_texts):].mean(axis=0).A1

    # Compute difference to find specific words
    tfidf_diff = target_class_tfidf - other_class_tfidf
    tfidf_scores_df = pd.DataFrame({'word': feature_names, 'tfidf_diff': tfidf_diff})
    tfidf_scores_df = tfidf_scores_df.sort_values(by='tfidf_diff', ascending=False).head(top_n)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tfidf_diff', y='word', data=tfidf_scores_df, palette="viridis")
    plt.title(f"Top {top_n} Words Specific to Class: {target_class.capitalize()}")
    plt.xlabel("TF-IDF Difference")
    plt.ylabel("Words")
    plt.show()


# Function to run the analysis for specific classes
def analyze_class_specific_words(df, top_n=10):
    classes = ['anxiety', 'depression', 'both']
    for target_class in classes:
        plot_class_specific_words(df, target_class, top_n)


# Heatmap of class similarities
def plot_class_similarity_heatmap(df):
    class_texts = df.groupby("Target")["Text"].apply(" ".join)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(class_texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=class_texts.index,
                yticklabels=class_texts.index)
    plt.title("Class Similarity Heatmap")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()


def plot_word_frequencies(normal_texts, top_n=20):
    """Generate a bar graph of the most frequently used words in normal entries."""
    # Use CountVectorizer to count word frequencies
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(normal_texts)

    # Sum up word frequencies and map them to their corresponding words
    word_sum = word_counts.sum(axis=0)
    words_freq = [(word, word_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # Select the top N words
    top_words = words_freq[:top_n]

    # Create a DataFrame for easy plotting
    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=words_df, palette='viridis')
    plt.title(f"Top {top_n} Most Frequently Used Words in Normal Entries", fontsize=16)
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Words", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_proposition_probabilities(normal_texts, model, class_labels):
    """Generate a bar graph showing probabilities of normal entries being related to anxiety or depression."""
    # Predict probabilities using the trained model
    probabilities = model.predict_proba(normal_texts)

    # Map probabilities to class labels
    probabilities_df = pd.DataFrame(probabilities, columns=class_labels)

    # Calculate average probabilities for anxiety and depression
    avg_probs = probabilities_df.mean().sort_values(ascending=False)

    # Create a bar graph for average probabilities
    plt.figure(figsize=(8, 6))
    sns.barplot(x=avg_probs.values, y=avg_probs.index, palette='mako')
    plt.title("Average Probabilities of Normal Entries Related to Anxiety or Depression", fontsize=16)
    plt.xlabel("Probability", fontsize=14)
    plt.ylabel("Classes", fontsize=14)
    plt.tight_layout()
    plt.show()


def preprocess_text(texts):
    """Apply consistent preprocessing to the input texts."""
    return texts.str.lower()


def analyze_normal_entries():
    # Load the dataset
    data = pd.read_csv("../data/balanced_dataset_50000_with_ids.csv")

    # Filter for entries labeled as "normal"
    normal_data = data[data['Target'] == 'normal']

    # Preprocess text
    normal_data['Text'] = preprocess_text(normal_data['Text'])

    # Load the saved model
    model = joblib.load("../persistency/multiclass_classifiers/model_3_class_classifier_random_forests.joblib")

    # Plot word frequencies for normal entries
    plot_word_frequencies(normal_data['Text'])

    # Plot probabilities of normal entries being related to anxiety or depression
    plot_proposition_probabilities(normal_data['Text'], model, class_labels=model.classes_)


# Call the function to analyze normal entries
# analyze_normal_entries()


def main():
    # file_path = "../data/balanced_dataset_50000_with_ids.csv"  # Replace with your file path
    # data = pd.read_csv(file_path)
    data = pd.read_csv("../data/BalancedDataset.csv")
    data = data.dropna(subset=['Text'])

    # data = data.drop(columns=['ID'])
    #
    # data = data.drop_duplicates(keep='first')
    # data.insert(0, 'ID', data.index)

    # depression_rows = data[data['Target'] == 'depression']
    # anxiety_rows = data[data['Target'] == 'anxiety']
    #
    # depression_rows_to_delete = depression_rows.sample(n=4980, random_state=42)
    # anxiety_rows_to_delete = anxiety_rows.sample(n=4200, random_state=42)
    # data = data.drop(depression_rows_to_delete.index)
    # data = data.drop(anxiety_rows_to_delete.index)
    # data.to_csv(file_path, index=False)

    print(data.describe())
    print(data.info())

    class_distribution = data['Target'].value_counts()

    print("Class Distribution:")
    print(class_distribution)

    plot_class_distribution_bar(data, 'Target')
    plot_class_distribution_pie(data, 'Target')

    plot_text_length_distribution(data, 'Text', 'Target')

    analyze_class_specific_words(data)

    plot_class_similarity_heatmap(data)


def main2():
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

    # Load the pretrained classifier
    classifier_t = joblib.load(
        '../persistency/multiclass_classifiers/tests_on_diff_datasets/model_3_class_classifier_random_forests_best_new_dataset_trained.joblib')

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')

    # Fit and transform the test list
    test_embeddings = tfidf_vectorizer.fit_transform(test_list).toarray()

    # Make predictions
    predictions = classifier_t.predict(test_embeddings)

    # Map numeric predictions back to labels
    labels = ["No Anxiety", "Anxiety"]
    results = [labels[pred] for pred in predictions]

    # Print the results
    for sentence, result in zip(test_list, results):
        print(f"Sentence: \"{sentence}\" => Prediction: {result}")


from sklearn.utils import shuffle


def make_dataset():
    save_path = "../data/BalancedDatasetShuffled.csv"
    df = pd.read_csv("../data/BalancedDataset.csv")
    df = shuffle(df)
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved to {save_path}")


if __name__ == '__main__':
    # make_dataset()
    # main()
    main2()
    print("dd")
