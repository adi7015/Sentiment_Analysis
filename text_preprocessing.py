import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
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

dataset = pd.read_csv('twitter_new_dataset.csv')
X = dataset.drop('cleaned_sentiment',axis =1)
y = dataset['cleaned_sentiment']

y.value_counts()

import tensorflow as tf

tf.__version__



from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

### Vocabulary size
voc_size=5000

messages=X.copy()

messages['tweet'][1]

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
corpus = []
for i in range(0,len(messages)):
    tweet = re.sub(TEXT_CLEANING_RE,' ', messages["tweet"][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ss = SnowballStemmer('english')
    tweet = [ss.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    
  ##############################################################################################  

#############################################################################################

corpus_df = DataFrame(corpus,columns = ['tweet'])

y_df = y.to_frame()




dataset_cleaned =corpus_df.join(y_df)

dataset_cleaned.to_csv('twitter_cleaned.csv')