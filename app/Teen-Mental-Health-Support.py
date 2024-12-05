import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# Load pre-trained models
depression_model = joblib.load('./persistency/classifiers/all-MiniLM-L6-v2_depression_classifier.joblib')
anxiety_model = joblib.load('./persistency/classifiers/all-MiniLM-L6-v2_anxiety_classifier.joblib')

# Initialize session state variables
if "posts" not in st.session_state:
    st.session_state.posts = [
        {"id": 1, "content": "Explored a beautiful park today!", "comments": []},
        {"id": 2, "content": "Feeling overwhelmed by work deadlines.", "comments": []},
        {"id": 3, "content": "Weekend plans are exciting!", "comments": []},
    ]

if "notification" not in st.session_state:
    st.session_state.notification = None  # No notification initially

# Function to predict sentiment
def predict_sentiment(comment):
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    comment_embedding = transformer_model.encode([comment])

    depression_prediction = depression_model.predict(comment_embedding)
    anxiety_prediction = anxiety_model.predict(comment_embedding)

    if depression_prediction[0] == "Depression":
        return "Depression detected"
    elif anxiety_prediction[0] == "Anxiety":
        return "Anxiety detected"
    return "No issues detected"

# Main app interface
st.title("Social Media Post Simulator")
st.subheader("Interact with posts and get insights!")

# Display notification bar
if st.session_state.notification:
    st.warning(st.session_state.notification)
    quiz_prompt = st.button("Take Quiz?")
    if quiz_prompt:
        st.info("Quiz functionality is under development!")  # Placeholder for quiz integration

# Display posts and comments
for post in st.session_state.posts:
    st.markdown(f"### Post {post['id']}: {post['content']}")
    st.write("*Comments:*")
    if not post['comments']:
        st.write("No comments yet.")
    else:
        for comment in post['comments']:
            st.write(f"- {comment}")

    # Add a comment
    with st.form(f"comment_form_{post['id']}"):
        comment_input = st.text_input(f"Add a comment to Post {post['id']}:")
        submit_button = st.form_submit_button("Submit")

        if submit_button and comment_input.strip():
            # Append the comment to the post
            post['comments'].append(comment_input.strip())

            # Analyze the comment
            result = predict_sentiment(comment_input.strip())

            # Update the notification bar if an issue is detected
            if result != "No issues detected":
                st.session_state.notification = f"⚠ {result}. Would you like to take a quiz?"

            # Refresh the comment field (clears the input)
            st.session_state[f"comment_input_{post['id']}"] = ""

# Reset notification button
if st.session_state.notification and st.button("Clear Notification"):
    st.session_state.notification = None