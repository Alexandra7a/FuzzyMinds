import streamlit as st


# Simulated function for sentiment analysis
def analyze_sentiment(text):
    anxiety_keywords = ["worried", "anxious", "nervous", "stressed"]
    depression_keywords = ["sad", "depressed", "hopeless", "tired"]

    if any(word in text for word in anxiety_keywords):
        return "Detected signs of anxiety."
    elif any(word in text for word in depression_keywords):
        return "Detected signs of depression."
    else:
        return "No significant signs of anxiety or depression detected."


# Chatbot response simulation
def chatbot_response(user_input):
    responses = {
        "hello": "Hi! How are you feeling today?",
        "how are you?": "I'm just a bot, but I'm here to listen!",
        "i feel sad": "I'm sorry to hear that. Do you want to talk more about it?",
        "i feel anxious": "That sounds tough. Try taking deep breaths. Do you want to chat?",
        "bye": "Goodbye! Take care."
    }
    for key in responses:
        if key in user_input.lower():
            return responses[key]
    return "I'm here to help. Tell me more."


# Combined function for sentiment analysis and chatbot
def analyze_and_respond(user_input):
    sentiment_result = analyze_sentiment(user_input)
    bot_reply = chatbot_response(user_input)
    return f"Sentiment Analysis: {sentiment_result}\n Chatbot: {bot_reply}"


# Streamlit interface
st.title("Teen Mental Health Support")

# Displaying project details
# st.subheader("Scop")
# st.write("Anxiety/Depression for teenagers - To enhance mental health support for teenagers by developing digital tools that can proactively identify signs of anxiety and depression. These tools aim to engage with adolescents in their digital environments—whether through chatbots, social media, video games, or other innovative platforms—to provide early intervention and emotional support.")
#
# st.subheader("Ideea de baza")
# st.write("The Early Discovery of Anxiety/Depression in Teenagers solution leverages digital platforms to identify and monitor mental health challenges among adolescents. The system combines multiple approaches, including AI-powered chatbots capable of conversational analysis, sentiment evaluation through social media interactions, and mental health assessments embedded in video game experiences. By engaging teenagers where they spend most of their time—whether online or gaming—the system aims to provide real-time insights and early warnings of anxiety or depression. This integrated solution offers a holistic approach, blending digital engagement with predictive analytics and personalized intervention strategies.")

# User input for sentiment analysis and chatbot
user_input = st.text_input("How are you feeling today?")

if user_input:
    result = analyze_and_respond(user_input)
    st.write(result)
