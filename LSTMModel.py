from keras.preprocessing.text import Tokenizer
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import numpy as np
from preprocessing import preprocess
import json
from sklearn.metrics import mean_squared_error
import math
import re
import os
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import settings

mask_synonyms = settings.mask_synonyms_list
tokenizer = Tokenizer(num_words=3500, split=' ')

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} ended!")

class LSTMModel():
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None, aspect_train=None, aspect_test=None, n_outputs=5):
        self.max_len = 10
        self.n_outputs = n_outputs

        self.x_train_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in x_train]
        self.x_test_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in x_test]

        self.aspect_train = aspect_train
        self.aspect_test = aspect_test

        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

        tokenizer.fit_on_texts(self.x_train_text)

        x_train = tokenizer.texts_to_sequences(self.x_train_text)
        self.x_train = pad_sequences(x_train, maxlen=self.max_len)

        x_test = tokenizer.texts_to_sequences(self.x_test_text)
        self.x_test = pad_sequences(x_test, maxlen=self.max_len)

        self.set_architecture()

    def set_architecture(self, lr=0.006, beta1=0.9, beta2=0.999, epsilon=1e-08):
        '''
        Method sets the architecture of the LSTM model
        :return: None
        '''
        self.model = Sequential()
        self.model.add(Embedding(3500, 128, input_length=self.max_len))
        self.model.add(LSTM(700, dropout=0.45))
        self.model.add(Dense(1, activation='tanh'))

        self.model.summary()

        adam = Adam(lr, beta1, beta2, epsilon)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=[keras.metrics.RootMeanSquaredError()])

    @staticmethod
    def tune_model(hp):
        '''
        Method to tune the model with keras-tuner
        :param hp: hyperparameters
        :return: None
        '''
        model = Sequential()
        model.add(Embedding(3500, 128, input_length=10)) #input_length=10
        model.add(LSTM(700, dropout=hp.Int("dropout_(divide_by_100)", min_value=45, max_value=55, step=5)/100)) #lstm_hidden=300
        model.add(Dense(1, activation='tanh'))

        lr = 0.006
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        adam = Adam(lr, beta1, beta2, epsilon)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=[keras.metrics.RootMeanSquaredError()])
        return model

    def one_hot_encoding(self, y):
        '''
        :param y: vector of numbers
        :return: one-hot-encoded array of numbers
        '''
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
        '''
        :param y: one-hot-encoded vector of numbers
        :return: vector of numbers
        '''
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

    def fit(self, epochs=30, verbose=2, batch_size=32, checkpoint_path=None):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        if checkpoint_path is not None:
            self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, callbacks=[cp_callback]) # callbacks=[CustomCallback()]
        else:
            self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

    # Evaluation
    def evaluate(self):
        # region CALCULATE_TRAIN_RMSE
        y_model = []
        for i, x in enumerate(self.x_train_text):
            y_model.append(self.predict(x))
        y_model = np.array(y_model)
        trainScore = math.sqrt(mean_squared_error(y_model, self.y_train))
        print('Train Score: %.5f RMSE' % (trainScore))
        # endregion

        # region CALCULATE_RANDOM_RMSE
        y_model_ = [0 for _ in enumerate(self.x_train_text)]
        trainScore = math.sqrt(mean_squared_error(y_model_, self.y_train))
        print('Random Score: %.5f RMSE' % (trainScore))
        # endregion

        # region CALCULATE_VALIDATION_RMSE
        self.model.evaluate(self.x_test, self.y_test)
        # endregion

    # Load weights
    def load_weights(self, checkpoint_path):
        '''
        Loads weights of the trained model
        :return: None
        '''
        latest_cp_file = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
        self.model.load_weights(latest_cp_file)

    # Testing
    def predict(self, x):
        '''
        :param x: input to a neural network
        :return: prediction of the neural network
        '''
        x = tokenizer.texts_to_sequences([x])
        x = pad_sequences(x, maxlen=self.max_len)
        y = self.model.predict(x)[0]
        print(y)
        return y




