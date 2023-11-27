import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')

ps = PorterStemmer()


def remove_website_links2(text):
    # Define regular expression patterns for different link types
    website_pattern = r"https?://[^\s]+"
    email_pattern = r"([a-zA-Z0-9_\.-]+@[a-zA-Z0-9_\.-]+\.[a-zA-Z]{2,6})"
    social_media_pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com|twitter\.com|facebook\.com|instagram\.com|linkedin\.com)"

    # Create a placeholder string to replace URLs
    placeholder = ""

    # Filter out URLs using regular expressions
    filtered_text = text
    for pattern in [website_pattern, email_pattern, social_media_pattern]:
        # Replace links with the placeholder
        filtered_text = re.sub(pattern, placeholder, filtered_text)

        # Handle cases where URLs are enclosed in quotation marks
        filtered_text = re.sub(r"\"" + placeholder + r"\"", "", filtered_text)
        filtered_text = re.sub(r"'" + placeholder + r"'", "", filtered_text)

    return filtered_text

def transform_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove website links using remove_website_links2 function
    text = remove_website_links2(text)

    # Tokenize the text using NLTK
    tokens = nltk.word_tokenize(text)

    # Filter alphanumeric characters
    y = []
    for token in tokens:
        if token.isalnum():
            y.append(token)

    # Remove stop words and punctuation
    filtered_tokens = []
    for token in y:
        if token not in stopwords and token not in string.punctuation:
            filtered_tokens.append(token)

    # Stem the remaining words
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(ps.stem(token))

    # Join the processed words
    processed_text = " ".join(stemmed_tokens)

    return processed_text


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

from PIL import Image
image = Image.open('img.png')
st.image(image, caption='Disaster Tweets')

# Project information section
with st.sidebar:
    st.title('Disaster Tweets')
    st.markdown("""
    ## Project Overview
    This project aims to develop a machine learning model for classifying disaster tweets. The model is trained on a dataset of tweets labeled as either disaster or non-disaster. The model can then be used to classify new tweets and identify potential disaster situations.

    ## Data and Methodology
    The dataset used for training the model was collected from Twitter using the Twitter API. The dataset includes tweets related to various natural disasters, including hurricanes, earthquakes, and floods. Each tweet was manually labeled as either disaster or non-disaster.
    The model was trained using a support vector machine (SVM) classifier. SVM is a widely used classification algorithm that is effective for text classification tasks.

    ## Results
    The model was able to achieve an accuracy of 92% on the test dataset. This suggests that the model is effective for classifying disaster tweets.

    ## Conclusion
    This project demonstrates the potential of machine learning to classify disaster tweets. The model can be used to identify potential disaster situations and help emergency responders coordinate their efforts.

    ## Future Work
    Future work could focus on improving the model's accuracy by using larger and more diverse datasets. Additionally, the model could be integrated with a real-time Twitter stream to identify disaster tweets as they occur.
    """)

st.title('Disaster Tweets')
input_Tweet = st.text_area("Enter Tweet ")

if st.button("predict"):
    

    #text preprocessing

    transformed_text = transform_text(input_Tweet)

    # vectorize

    vector_input = tfidf.transform([transformed_text])

    # prediction 
    result = model.predict(vector_input)[0]

    # disply
    if result == 1:
        st.header("Disaster Tweet")
    else:
        st.header("Not Disaster Tweet")
