import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from LSTMModel import LSTMModel
from keras_preprocessing.text import Tokenizer
import preprocessing
import kerastuner as kt
from kerastuner import RandomSearch, Objective
from kerastuner.engine.hyperparameters import HyperParameters
from preprocessing import preprocess
import numpy as np
from sklearn.metrics import mean_squared_error
import time
import tensorflow as tf
import os

def test_model(country, checkpoint_path, model, train):
    # region LOAD_MODEL
    if not train:
        model.load_weights(checkpoint_path)
    # endregion

    predict_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\IEEE_Dataset\refactored_datasets\new_' + country + '.csv'
    predict_dataset_path = re.sub(r'\\', '/', predict_dataset_path)
    predict_dataset_path_save = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\IEEE_Dataset\refactored_datasets\v2\new_' + country + '_sent5_v2.csv'
    predict_dataset_path_save = re.sub(r'\\', '/', predict_dataset_path_save)

    # region PREDICTION
    predict_dataset = pd.read_csv(predict_dataset_path)
    y = {}
    y['model_output'] = []

    predict_text = predict_dataset.raw_text
    for text in predict_text:
        y['model_output'].append(model.predict(text))
    print(f"END {country}!!!")
    # endregion

    # region SAVING RESULTS(deactivated)
    predict_dataset['model_output'] = y['model_output']
    predict_dataset.to_csv(predict_dataset_path_save, index=False)
    # endregion

def train_model(checkpoint_path=None, train = True, validate = True, tune = False):
    # region LOAD DATA FROM OS
    train_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\IEEE_Dataset\refactored_datasets\train_dataset.csv'
    train_dataset_path = re.sub(r'\\', '/', train_dataset_path)
    validation_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\IEEE_Dataset\refactored_datasets\validation_dataset.csv'
    validation_dataset_path = re.sub(r'\\', '/', validation_dataset_path)

    train_set = pd.read_csv(train_dataset_path)
    test_set = pd.read_csv(validation_dataset_path)
    # endregion

    #region INSTANTIATE TRAIN AND TEST DATA
    train_text = train_set.raw_text
    train_labels = train_set.label
    train_aspects = train_set.aspects

    test_text = test_set.raw_text
    test_labels = test_set.label
    test_aspects = test_set.aspects
    #endregion

    # region FIT THE MODEL
    model = LSTMModel(x_train=train_text, y_train=train_labels, x_test=test_text, y_test=test_labels, aspect_train=train_aspects, aspect_test=test_aspects)
    if train:
        model.fit(epochs=30, checkpoint_path=checkpoint_path)
    # endregion

    # region EVALUATE THE MODEL (validation actually)
    if validate:
        model.evaluate()
    # endregion

    # region TUNE_MODEL_WITH_KERAS_TUNER (deactivated)
    if tune:
        LOG_DIR = f'{int(time.time())}'
        LOG_DIR = r"C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\KerasTunerModels"
        # Beware that the max_trials is the sum of the number you put here and the experiments already performed
        tuner = RandomSearch(
                LSTMModel.tune_model,
                objective=kt.Objective('val_root_mean_squared_error', direction='min'),
                max_trials=150,
                executions_per_trial=2,
                directory=os.path.normpath('C:/'),
                project_name='txmm_LSTM_tunning_2'
            )

        # region KEEP_ME_SMALL_(IM JUST HANDLING DATA)
        tokenizer = Tokenizer(num_words=3500, split=' ')

        x_train_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in train_text]
        x_test_text = [preprocess(tweet, remove_stopwords=False, pos_tagging=False, join_text=True) for tweet in test_text]
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        tokenizer.fit_on_texts(x_train_text)
        x_train = tokenizer.texts_to_sequences(x_train_text)
        x_train = pad_sequences(x_train, maxlen=10)
        x_test = tokenizer.texts_to_sequences(x_test_text)
        x_test = pad_sequences(x_test, maxlen=10)
        #endregion

        tuner.search(x=x_train, y=y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))
        # endregion

    return model

if __name__ == '__main__':
    train = False
    checkpoint_path = 'model_checkpoint/cp.ckpt'
    model = None

    if checkpoint_path == '':
        checkpoint_path = None
    model = train_model(checkpoint_path=checkpoint_path, train=train)
    countries = ['nigeria', 'united_kingdom', 'united_states', 'australia', 'india', 'canada']
    countries = ['nigeria']
    for country in countries:
        test_model(country, checkpoint_path, model=model, train=train)