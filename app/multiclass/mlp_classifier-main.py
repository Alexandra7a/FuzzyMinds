import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.version_utils import callbacks


def make_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.show()


def model(X_train, X_test, y_train, y_test):
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    model = keras.Sequential([layers.Dense(units=100, activation='relu', input_shape=[384]),
                              #layers.Dropout(rate=0.3),
                              layers.Dense(units=60, activation='relu'),
                              layers.Dense(units=4, activation='softmax'),
                              ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=256,

        epochs=100,
        callbacks=[early_stopping],  # put your callbacks in a list
        verbose=0,  # turn off training log
    )

    # plotting
    print("GRAFIC 1")

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot();
    print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

    print("GRAFIC 2")
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_pred_prob = model.predict(X_test)  # Probabilities
    y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted classes

    # Step 6: Calculate and display metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(3)]))

    model.save('../persistency/multiclass_classifiers/model_3_class_classifier.h5')


# Class distribution as pie chart
def plot_class_distribution_pie(df, class_column):
    class_counts = df[class_column].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("viridis", len(class_counts)))
    plt.title("Class Distribution (Pie Chart)")
    plt.show()


def make_model():
    data = pd.read_csv("../data/balanced_dataset_50000_with_ids.csv")
    # data = data[:20000]
    data['Target'] = data['Target'].apply(lambda x:
                                          0 if x == "normal" else
                                          1 if x == "anxiety" else
                                          2)

    # data.dropna(inplace=True)
    # data.drop_duplicates(subset=["Text","Target"],keep='first', inplace=True)
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle entire dataset

    print(data.head(20))
    plot_class_distribution_pie(data, 'Target')
    print(data.describe())

    X = data['Text']
    y = data['Target']

    # Encode the text using a transformer
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = transformer_model.encode(X.tolist())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.25, random_state=42, stratify=y)

    print("Training set class distribution:")
    print(pd.Series(y_train).value_counts())
    print("Test set class distribution:")
    print(pd.Series(y_test).value_counts())
    # Call the model with tracking function
    model(X_train, X_test, y_train, y_test)


def main():
    start_time = time.time()
    make_model()
    end_time = time.time()
    print("Total Time:", end_time - start_time)


if __name__ == '__main__':
    # print(tf.__version__)
    main()
