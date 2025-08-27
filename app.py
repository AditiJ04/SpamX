import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Load saved model, vectorizer, and label encoder
model = pickle.load(open('spam_model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Title of the app
st.title("Spam Email Detection")
st.write("Type any email or message below to check if it's Spam or Ham.")

# Text input
user_input = st.text_area("Enter your message here:")

# Button to predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
        # Preprocess input
        message = re.sub('[^a-zA-Z]', ' ', user_input)
        message = message.lower().split()
        message = [ps.stem(word) for word in message if word not in stop_words]
        message = ' '.join(message)
        
        # Vectorize and predict
        message_vectorized = cv.transform([message]).toarray()
        prediction = model.predict(message_vectorized)
        prediction_label = le.inverse_transform(prediction)[0]
        
        # Display result
        if prediction_label.lower() == "spam":
            st.error(f"The message is classified as: {prediction_label}")
        else:
            st.success(f"The message is classified as: {prediction_label}")
            
            

