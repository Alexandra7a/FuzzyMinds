# Early Discovery of Anxiety and Depression in Teenagers Using Digital Tools

## Overview

This project leverages advanced artificial intelligence and machine learning methods to identify and monitor signs of anxiety and depression in teenagers. By integrating predictive analytics with digital engagement platforms like chatbots, social media, and video games, the system aims to provide early interventions and support for adolescent mental health challenges.

## Objective

The primary goal is to enhance mental health support for teenagers by:
- Detecting early signs of anxiety and depression.
- Reducing barriers to seeking help by integrating support systems into familiar digital environments.
- Providing timely, personalized interventions through AI-driven tools.

## Key Features

- **AI-Powered Chatbots**: Enable conversational analysis and real-time emotional assessment.
- **Sentiment Analysis**: Analyze social media interactions to detect emotional distress.
- **Game-Based Assessments**: Integrate mental health evaluations into video games to engage teenagers in non-intrusive ways.

## Datasets Used

1. **Reddit Mental Health Dataset**: Posts labeled with mental health conditions.
2. **Emotion Recognition Dataset**: Text samples labeled with emotional states.
3. Combined and preprocessed datasets ensure balanced representation across Anxiety, Depression, and Normal categories.

## Algorithms and Models

- **Support Vector Machines (SVM)**
- **Linear Regression**
- **Decision Trees**
- **Neural Networks with Transformers (e.g., Sentence Transformers)**
- **Random Forest Classifiers**

### Experimental Results
- **Small Data Experiments**: Achieved moderate accuracy (e.g., SVM with Word2Vec embeddings yielded ~64% accuracy).
- **Large Data Experiments**: Higher accuracy with advanced methods like Multi-layer Perceptron Classifiers (~93% accuracy).
- **Final Approach**: Random Forest Classifier achieved 92% accuracy in multi-class classification of Anxiety, Depression, and Normal states.

## Technical Details

- **Text Representations**: TF-IDF Vectorization, Transformer-based Sentence Embeddings.
- **Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Code Efficiency**:
  - Spatial Complexity: Optimized for large datasets using sparse matrix techniques.
  - Temporal Complexity: Fine-tuned hyperparameters and efficient cross-validation for scalable performance.

## Limitations

- High computational requirements for transformer-based models.
- Lack of real-time validation and generalizability across cultural and linguistic contexts.
- Sensitivity to ambiguous emotional signals.

## Future Improvements

- **Multimodal Data Integration**: Incorporate voice tone, facial expressions, and physiological data.
- **Real-Time Adaptation**: Enable continuous learning and dynamic model updates.
- **Global Applicability**: Expand datasets to include diverse linguistic and cultural expressions of mental health.

