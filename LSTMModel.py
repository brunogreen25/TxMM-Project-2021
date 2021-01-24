from keras.preprocessing.text import Tokenizer
import keras
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import numpy as np
from preprocessing import preprocess
import json
import re

mask_synonyms = ['mask', 'masks', 'wearamask', 'wear a mask', 'wearmask', 'wear mask', 'face shield', 'faceshield', 'masks4all', 'mask for all', 'masks for all']
tokenizer = Tokenizer(num_words=3500, split=' ')

class LSTMModel():
    def __init__(self, x_train, x_test, y_train, y_test, aspect_train, aspect_test, n_outputs=5, x_prediction=None):
        self.x_train_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in x_train]
        self.x_test_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in x_test]

        self.aspect_train = aspect_train
        self.aspect_test = aspect_test

        #self.x_train_text = self.add_aspect(x_train_text, aspect_train)
        #self.x_test_text = self.add_aspect(x_test_text, aspect_test)

        self.max_len = 10
        self.n_outputs = n_outputs

        self.y_train = self.one_hot_encoding(y_train)
        self.y_test = self.one_hot_encoding(y_test)

        tokenizer.fit_on_texts(self.x_train_text)

        x_train = tokenizer.texts_to_sequences(self.x_train_text)
        self.x_train = pad_sequences(x_train, maxlen=self.max_len)

        x_test = tokenizer.texts_to_sequences(self.x_test_text)
        self.x_test = pad_sequences(x_test, maxlen=self.max_len)

        self.set_architecture()

    def set_architecture(self):
        self.model = Sequential()
        self.model.add(Embedding(3500, 128, input_length=self.max_len))
        self.model.add(LSTM(300, dropout=0.2))
        self.model.add(Dense(self.n_outputs, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    def one_hot_encoding(self, y):
        labels_ = []
        n_bins = self.n_outputs
        for label in y:
            encoded_label = np.zeros(n_bins)
            borders = np.arange(-1, 1+2/n_bins, 2/n_bins)
            for i in range(n_bins):
                if label >= borders[i] and label <= borders[i+1]:
                    encoded_label[i] = 1
                    break

            labels_.append(encoded_label)
        return np.array(labels_)

    def one_hot_decoding(self, y):
        labels = []
        n_bins = self.n_outputs
        for label in y:
            index = np.where(label == 1)
            if any(label):
                labels.append(index[0][0])
            else:
                median = int(label.size/2)
                labels.append(median)
        labels = np.array(labels)
        return labels

    def fit(self, epochs=30, verbose=2, batch_size=32):
        self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def evaluate(self):
        # Accuracy
        self.model.evaluate(self.x_test, self.y_test)

        # Confusion matrix
        y_model = []
        for i, x in enumerate(self.x_test_text):
            #x = self.add_aspect(x, self.aspect_test[i])
            y = self.predict(x)
            y_model.append(y)

        y_model_ = [y[0] for y in y_model]
        y_model = np.argmax(y_model_, axis=1)
        y_test = self.one_hot_decoding(self.y_test)
        return classification_report(y_test, y_model)

    def predict(self, x):
        x = tokenizer.texts_to_sequences([x])
        x = pad_sequences(x, maxlen=self.max_len)
        y = self.model.predict(x)
        return y



