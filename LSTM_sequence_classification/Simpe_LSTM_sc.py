# -*- coding: utf-8 -*-
# Aim: sequence classification via LSTM

# Specific: We will map each word onto a 32 length real valued vector. We will also limit the total number of words that we are interested in modeling to the 5000 most frequent words, and zero out the rest. Finally, the sequence length (number of words) in each review varies, so we will constrain each review to be 500 words, truncating long reviews and pad the shorter reviews with zero values.

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(7)

# We need to load the IMDB dataset. We are constraining the dataset to the top frequent 5,000 words. We also split the dataset into train (50%) and test (50%) sets.
top_words = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
# Input length: 500
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, 
	validation_data=(X_test, y_test),
	nb_epoch=3, batch_size=64)

# evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Running result
# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to                     
# ====================================================================================================
# embedding_1 (Embedding)          (None, 500, 32)       160000      embedding_input_1[0][0]          
# ____________________________________________________________________________________________________
# lstm_1 (LSTM)                    (None, 100)           53200       embedding_1[0][0]                
# ____________________________________________________________________________________________________
# dense_1 (Dense)                  (None, 1)             101         lstm_1[0][0]                     
# ====================================================================================================
# Total params: 213301
# ____________________________________________________________________________________________________
# None
# Train on 25000 samples, validate on 25000 samples
# Epoch 1/3
# 25000/25000 [==============================] - 444s - loss: 0.4909 - acc: 0.7520 - val_loss: 0.3540 - val_acc: 0.8552
# Epoch 2/3
# 25000/25000 [==============================] - 451s - loss: 0.2977 - acc: 0.8808 - val_loss: 0.3250 - val_acc: 0.8614
# Epoch 3/3
# 25000/25000 [==============================] - 449s - loss: 0.2741 - acc: 0.8887 - val_loss: 0.3300 - val_acc: 0.8598
# Accuracy: 85.98%