import pandas as pd
import re
import aspect_based_sentiment_analysis
import preprocessing
import json

if __name__=='__main__':
    train_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\train_dataset.csv'
    train_dataset_path = re.sub(r'\\', '/', train_dataset_path)
    validation_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\validation_dataset.csv'
    validation_dataset_path = re.sub(r'\\', '/', validation_dataset_path)

    train_sample_n = 117
    validation_sample_n = 39
    sample_num = [train_sample_n, validation_sample_n]

    instance = {
        'tweet_id': ['***'],
        'username': ['***'],
        'user_followers_count': [0],
        'coordinates': [0],
        'datetime': ['***'],
        'location': ['***'],
        'raw_text': [''],
        'sentiment_score': ['0'],
        'label': [''],
        'aspects': ['']
    }
    dataset_field_names = ['tweet_id', 'username', 'user_followers_count', 'coordinates', 'datetime', 'location',
                           'raw_text', 'sentiment_score', 'aspect', 'label']

    dataset_paths = [train_dataset_path, validation_dataset_path]
    for i, dataset_path in enumerate(dataset_paths):
        df = pd.read_csv(dataset_path)
        new_df = pd.DataFrame()
        label=''

        # Add labels to the dataset
        for index in range(sample_num[i]):
            print(index+1,"="*50+"NEW TWEET"+"="*50)

            # Enter fake text
            text = str(input())

            # Print aspect from that text
            aspect = aspect_based_sentiment_analysis.get_aspect(text)
            json_aspect = json.dumps(aspect)
            json_aspect = re.sub(',', ';', json_aspect)
            print(aspect)

            # Enter label
            while not str(label).isdigit() or float(label)<0 or float(label)>100:
                label = input()
            label = float(label)
            label = (label - 50)/50
            label = str(round(label, 2))

            # Save text, aspect and label to the dataframe
            instance['raw_text'][0] = text
            instance['aspects'][0] = json_aspect
            instance['label'][0] = str(label)

            row = pd.DataFrame.from_dict(instance)
            df = df.append(row, ignore_index=True)

            print("=" * 110)

        # Save new dataset and delete rows from the old dataset (and save it as well)
        df.to_csv(dataset_paths[i], index=False)

        # Print progress
        if i==0:
            print("TRAIN DATASET LABELLING COMPLETED")
        else:
            print("VALIDATION DATASET LABELLING COMPLETED")