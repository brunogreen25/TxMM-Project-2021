import re
import pandas as pd
from LSTMModel import LSTMModel
from keras_preprocessing.text import Tokenizer
import preprocessing
from preprocessing import preprocess
import numpy as np


if __name__ == '__main__':
    train_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\train_dataset.csv'
    train_dataset_path = re.sub(r'\\', '/', train_dataset_path)
    validation_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\validation_dataset.csv'
    validation_dataset_path = re.sub(r'\\', '/', validation_dataset_path)
    country = 'nigeria'
    predict_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\new_'+country+'.csv'
    predict_dataset_path = re.sub(r'\\', '/', predict_dataset_path)
    predict_dataset_path_save = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\new_'+country+'_sent5.csv'
    predict_dataset_path_save = re.sub(r'\\', '/', predict_dataset_path_save)

    # FITTING THE MODEL
    train_set = pd.read_csv(train_dataset_path)
    test_set = pd.read_csv(validation_dataset_path)

    train_text = train_set.raw_text
    train_labels = train_set.label
    train_aspects = train_set.aspects

    test_text = test_set.raw_text
    test_labels = test_set.label
    test_aspects = test_set.aspects

    model = LSTMModel(x_train=train_text, y_train=train_labels, x_test=test_text, y_test=test_labels, aspect_train=train_aspects, aspect_test=test_aspects, x_prediction=['i hate wearing masks'])
    model.fit()
    result = model.evaluate()
    print(result)

    report = model.evaluate()
    print(report)

    predict_dataset = pd.read_csv(predict_dataset_path)
    y = {}
    y['model_output'] = []

    predict_text = predict_dataset.raw_text
    for text in predict_text:
        y['model_output'].append(model.predict(text))

    y_model_ = [y[0] for y in y['model_output']]
    y['model_output'] = np.argmax(y_model_, axis=1)
    print("END!!!")
    predict_dataset['model_output'] = y['model_output']
    predict_dataset.to_csv(predict_dataset_path_save, index=False)

    #y = model.predict(['i hate wearing mask'])
   # print(y)

