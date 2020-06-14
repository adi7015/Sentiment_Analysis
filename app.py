#creating the UI of model using strealit

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import streamlit as st
import numpy as np
import pickle


st.title("Twitter Sentiment Analysis")
st.write("This ML model will guess if a given tweet is positive or negative by using NLP. "
         "This model was trained using Tensorflow and was trained on the twitter sentiment analysis-dataset of twitter reviews.")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("model.h5")

tokenizer = pickle.load(open('token.pickle','rb'))






import re



sample_text = st.text_input("Enter any text")


if sample_text != '':
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z]|[0-9]+"
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    corpus = []
    
    tweet = re.sub(TEXT_CLEANING_RE,' ', sample_text)
      
    tweet = tweet.lower()
    tweet = tweet.split()
    ss = SnowballStemmer('english')
    tweet = [ss.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    
    
    
    
    
    max_len = 60
    
    
    
    with st.spinner("Tokenizing Text....."):
        
        sequences = tokenizer.texts_to_sequences(corpus)
        test_data = sequence.pad_sequences(sequences, maxlen = max_len)
    
    
    preds = loaded_model.predict(test_data)
    
    
    
    
    
    st.subheader("The given review was : ")
    
    if preds > [0.5]:
        st.write('Positive')
    else:
       st.write('Negative')
        
