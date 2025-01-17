import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# Load pre-trained models (placeholders for demo)
depression_model = joblib.load('../persistency/classifiers/all-MiniLM-L6-v2_depression_classifier.joblib')
anxiety_model = joblib.load('../persistency/classifiers/all-MiniLM-L6-v2_anxiety_classifier.joblib')

# Initialize session state variables
if "posts" not in st.session_state:
    st.session_state.posts = [
        {"id": 1, "content": "Explored a beautiful park today!", "comments": []},
        {"id": 2, "content": "Feeling overwhelmed by work deadlines.", "comments": []},
        {"id": 3, "content": "Weekend plans are exciting!", "comments": []},
    ]

if "notification" not in st.session_state:
    st.session_state.notification = None  # No notification initially

if "comment_inputs" not in st.session_state:
    st.session_state.comment_inputs = {post['id']: "" for post in st.session_state.posts}


# Function to predict sentiment
def predict_sentiment(comment):
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    comment_embedding = transformer_model.encode([comment])

    depression_prediction = depression_model.predict(comment_embedding)
    anxiety_prediction = anxiety_model.predict(comment_embedding)

    if depression_prediction[0] == 1:
        return "Depression detected"
    elif anxiety_prediction[0] == 1:
        return "Anxiety detected"
    return "No issues detected"


# CSS for the pop-up notification (positioned at bottom right, with "X" button)
def inject_css():
    st.markdown(
        """
        <style>
        .popup {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #000000;
            padding: 15px;
            border: 1px solid #ffffff;
            border-radius: 10px;
            z-index: 1000;
            width: 300px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .popup button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        .popup button:hover {
            background-color: #0056b3;
        }
        .popup .close-btn {
            position: absolute;
            top: 5px;
            right: 10px;
            background-color: transparent;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
        }
        .popup .close-btn:hover {
            color: red;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Call the CSS injection
inject_css()

# Display notification pop-up
if st.session_state.notification:
    st.markdown(
        f"""
        <div class="popup">
            <button class="close-btn" onclick="window.location.reload();">×</button>
            <strong>⚠ {st.session_state.notification}</strong>
            <br>
            <button onclick="window.alert('Quiz functionality is under development!')">Take Quiz</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # When X button is clicked, update session state to hide notification
    if st.button('Close Notification'):
        st.session_state.notification = None

# Main app interface
st.title("Social Media Post Simulator")
st.subheader("Interact with posts and get insights!")

# Display posts and comments
for post in st.session_state.posts:
    post_id = post['id']
    st.markdown(f"### Post {post_id}: {post['content']}")
    st.write("*Comments:*")
    if not post['comments']:
        st.write("No comments yet.")
    else:
        for comment in post['comments']:
            st.write(f"- {comment}")

    # Add a comment
    with st.form(f"comment_form_{post_id}") as form:
        comment_input = st.text_input(
            f"Add a comment to Post {post_id}:",
            value=st.session_state.comment_inputs[post_id]
        )
        submit_button = st.form_submit_button("Submit")

        if submit_button and comment_input.strip():
            # Append the comment to the post
            post['comments'].append(comment_input.strip())

            # Save the input in session state
            st.session_state.comment_inputs[post_id] = ""

            # Analyze the comment
            result = predict_sentiment(comment_input.strip())

            # Update the notification pop-up if an issue is detected
            if result != "No issues detected":
                st.session_state.notification = f"{result}. Would you like to take a quiz?"

# Clear the notification when the "Clear Notification" button is pressed
if st.session_state.notification and st.button("Clear Notification"):
    st.session_state.notification = None
