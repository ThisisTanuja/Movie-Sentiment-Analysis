#importing necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import keras

from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer



# Importing and Analyzing the Dataset
#the IMDB movie review dataset is imported using the pandas library
movie_reviews = pd.read_csv("IMDB Dataset.csv")

#isnull() is used to check for any missing values in the dataset
movie_reviews.isnull().values.any()
print(movie_reviews.shape)

#head() method is used to display the first few rows of the dataset
print(movie_reviews.head())
#print(movie_reviews["review"])

#countplot() of the seaborn library is used to visualize the distribution of the labels in the dataset
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
ax = sns.countplot(x='sentiment', data=movie_reviews)
ax.set_title('Distribution of Sentiments in Movie Reviews', fontsize='16')
ax.set_xlabel('Sentiment', fontsize='12')
ax.set_ylabel('Count', fontsize='12')
plt.show()


#data preprocessing
def preprocess_text(sen):
    # Removing html tags <> with empty space
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


TAG_RE = re.compile(r'<[^>]+>')

#remove_tags() function is used to remove HTML tags from the text
def remove_tags(text):
    return TAG_RE.sub('', text)

#the x variable is created to hold the preprocessed text data
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))
#print(X[3])

#the Y variable is created to hold the labels which are converted to binary values(0 for negative and 1 for positive)
Y = movie_reviews['sentiment']
Y = np.array(list(map(lambda x: 1 if x == "positive" else 0, Y)))

#train_test_split() function is used to split the data into training and testing sets
# 80% for the training set and 20% for the testing set.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


#Tokenizer class from the Keras library is used to create a word-to-index dictionary.
tokenizer = Tokenizer(num_words=5000)

#fit_on_texts() method is used to fit the tokenizer on the training data
tokenizer.fit_on_texts(X_train)

#texts_to_sequences() method is used to convert the text data into sequences of integers.
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
#pad_sequences() method is used to pad the sequences to a maximum length of 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# use GloVe embeddings to create our feature matrix(pre-trained word embeddings)
from numpy import array
from numpy import asarray
from numpy import zeros

#embeddings_dictionary is created to hold the embeddings for each word in the dataset
embeddings_dictionary = dict()

#the glove file is opened, and the embeddings for each word are extracted and added to the embeddings_dictionary
glove_file = open('glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

#embedding_matrix is created to hold the embeddings for each word in the tokenizer dictionary
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# embedding_matrix will contain 92547 rows (one for each word in the corpus)



        # DEEP LEARNING MODELS

        # 1  Text Classification with Simple Neural Network

#the Sequential() model from the keras library is used to create a neural network model
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping


# Define input shape
inputs = Input(shape=(maxlen,))

# Embedding layer with pre-trained embeddings
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)

# Flatten layer
flatten_layer = Flatten()(embedding_layer)

# Output layer with sigmoid activation
output_layer = Dense(1, activation='sigmoid')(flatten_layer)

# Create the model
model = Model(inputs=inputs, outputs=output_layer)

# Compile the model with appropriate optimizer, loss, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model using fit() method with early stopping and validation split
history = model.fit(X_train, Y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate the model on the testing data
score = model.evaluate(X_test, Y_test, verbose=1)

# Display the test score and accuracy
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


#plotting the results
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('SNN - model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('SNN - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#2 Text Classification with a Convolutional Neural Network(CNN)

from keras.layers.convolutional import Conv1D

# Define input shape
inputs = Input(shape=(maxlen,))

# Embedding layer with pre-trained embeddings
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)



#after the embedding layer, Conv1D() layer is added to the model which performs 1D convolution
#this is useful for processing sequential data like text
#128 parameter is the number of filters in the layer, 5 is the kernel size, relu activation function is used
conv1d_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
#GlobalMaxPooling1D() layer is added next to the model which takes maximum value of each feature map generated by the convolution layer
globalmaxpooling1d = GlobalMaxPooling1D()(conv1d_layer)
#finally, a dense() layer with a single neuron is added to the model and the sigmoid activation function is used
#this layer is used for binary classification, where the output should be a single value between 0 and 1.
output_layer = Dense(1, activation='sigmoid')(globalmaxpooling1d)
# Create the model
model = Model(inputs=inputs, outputs=output_layer)

# Compile the model with appropriate optimizer, loss, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model using fit() method with early stopping and validation split
history = model.fit(X_train, Y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate the model on the testing data
score = model.evaluate(X_test, Y_test, verbose=1)

# Display the test score and accuracy
print("Test Score:", score[0])
print("Test Accuracy:", score[1])



#plotting the results
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('CNN - model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('CNN - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# 3 Text Classification with Recurrent Neural Network (LSTM)
# Long Short Term Memory Network


#this code is using a RNN with a LSTM architecture
from tensorflow.keras.layers import LSTM

# Define input shape
inputs = Input(shape=(maxlen,))

# Embedding layer with pre-trained embeddings
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)

#LSTM layer is added next which takes the output of the embedding layer as input and learns the sequential relationships between words in the text data
#has 128 units, which determines the number of hidden states to be used in the model.
lstm = LSTM(128)(embedding_layer)

#Dense() layer is the final layer added with a sigmoid activation function and outputs a binary classification score
output_layer = Dense(1, activation='sigmoid')(lstm)
# Create the model
model = Model(inputs=inputs, outputs=output_layer)

# Compile the model with appropriate optimizer, loss, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model using fit() method with early stopping and validation split
history = model.fit(X_train, Y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate the model on the testing data
score = model.evaluate(X_test, Y_test, verbose=1)

# Display the test score and accuracy
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


#plotting the results
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('RNN - model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('RNN - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







