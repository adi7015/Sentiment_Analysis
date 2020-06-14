import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
import pickle
from tensorflow.keras.models import Model

df = pd.read_csv('twitter_cleaned.csv')



def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.tweet.values,
                                                    df.cleaned_sentiment.values,
                                                    stratify=df.cleaned_sentiment.values,
                                                    test_size = 0.2, random_state = 42,
                                                    shuffle = True)
X_train=X_train.astype(str)
X_test=X_test.astype(str)

# using keras tokenizer here
token = text.Tokenizer(num_words=None)

max_len = 60
token.fit_on_texts(list(X_train) + list(X_test))

import pickle

with open('token.pickle', 'wb') as handle:
     pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)




X_train_seq = token.texts_to_sequences(X_train)
X_test_seq = token.texts_to_sequences(X_test)

#zero pad the sequences
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)

word_index = token.word_index



# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open(r'C:\Users\adity\Downloads\glove.840B.300d\glove.840B.300d.txt',
        'r',encoding='utf-8' )
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        
# A simple LSTM with glove embeddings and one dense layer
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        

model.summary()

model.fit(X_train_pad, y_train, nb_epoch=15, batch_size=32)

scores = model.predict(X_test_pad)
print("Auc: %.2f%%" % (roc_auc(scores,y_test)))

import re
sample_text = 'i feel very happy and excited!'



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
    

    



sequences = token.texts_to_sequences(corpus)
test_data = sequence.pad_sequences(sequences, maxlen = max_len)

preds = model.predict(test_data)









if preds>[[0.5]]:
    print('positive')
else:
    print('negative')
    
    




from keras.models import model_from_json


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")