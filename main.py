from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import pandas as pd
import keras
from keras import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Flatten
from keras.preprocessing.text import Tokenizer
import numpy as np
from numpy import array, asarray, zeros

def remove_stop_words(tokens): 
    return [word for word in tokens if word not in stopwords.words('english')]

def lower_token(tokens): 
    return [w.lower() for w in tokens]  

data = pd.read_csv('./data/data.csv')
data.columns = ['Text', 'Label']
data.head() 

tokens = [word_tokenize(token) for token in data.Text] 
    
lower_tokens = [lower_token(token) for token in tokens]

filtered_words = [remove_stop_words(token) for token in lower_tokens]
data['Text_Final'] = [' '.join(token) for token in filtered_words]
data['Tokens'] = filtered_words

data = data[['Text_Final', 'Tokens', 'Label']]
data.head()

data_train, data_test = train_test_split(data, test_size=0.10)

###   SVM
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(data_train['Text_Final'])
train_svm_data = ngram_vectorizer.transform(data_train['Text_Final'])
test_svn_data = ngram_vectorizer.transform(data_test['Text_Final'])
svm_model = LinearSVC(C=0.05)
svm_model.fit(train_svm_data, data_train['Label'])
svm_predict = svm_model.predict(test_svn_data)
print("SVM confusion matrix=%s" % (confusion_matrix(data_test['Label'], svm_predict)))
print("Total SVM accuracy=%s" % (accuracy_score(data_test['Label'], svm_predict)))

###   CNN
tokenizer = Tokenizer(num_words=8635)
tokenizer.fit_on_texts(data_train['Text_Final'])
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

embeddings_dictionary = dict()
glove_file = open('./data/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

training_sequences = tokenizer.texts_to_sequences(data_train['Text_Final'])
testing_sequences = tokenizer.texts_to_sequences(data_test['Text_Final'])
train_cnn_data = keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=maxlen)
test_cnn_data = keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=maxlen)

history = model.fit(train_cnn_data, data_train['Label'], batch_size=128, epochs=6, verbose=1, validation_split=0.10)

predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

prediction_labels=[]
for p in predictions:
    prediction_labels.append(int(np.round(p)))
print("CNN confusion matrix=%s" % (confusion_matrix(data_test['Label'], prediction_labels)))
print("Total CNN accuracy=%s" % (accuracy_score(data_test['Label'], prediction_labels)))
