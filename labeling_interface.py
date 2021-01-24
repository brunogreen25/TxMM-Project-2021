import pandas as pd
import re

if __name__=='__main__':
    train_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\train_dataset.csv'
    train_dataset_path = re.sub(r'\\', '/', train_dataset_path)
    validation_dataset_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\validation_dataset.csv'
    validation_dataset_path = re.sub(r'\\', '/', validation_dataset_path)

    train_sample_n = 197
    validation_sample_n = 99
    sample_num = [train_sample_n, validation_sample_n]

    dataset_to_extract_from_p = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset\refactored_datasets\new_united_states.csv'
    dataset_to_extract_from_p = re.sub(r'\\', '/', dataset_to_extract_from_p)

    dataset_field_names = ['tweet_id', 'username', 'user_followers_count', 'coordinates', 'datetime', 'location',
                           'raw_text', 'sentiment_score', 'aspect', 'label']

    datasets = [train_dataset_path, validation_dataset_path]
    for i, dataset_path in enumerate(datasets):
        df = pd.DataFrame(columns=dataset_field_names)
        df_to_extract_from = pd.read_csv(dataset_to_extract_from_p)
        counter = 1
        label=''

        # Add labels to the dataset
        for index, row in df_to_extract_from.sample(sample_num[i]).iterrows():
            print(str(counter) + ') ' + row['raw_text'])
            counter += 1
            while not str(label).isdigit():
                label = input()
            label = float(label)
            label = (label - 50)/50

            # Save index of row and label
            row['label'] = round(label, 2)
            df = df.append(row, ignore_index=True)
            df_to_extract_from = df_to_extract_from.drop(index)

        # Save new dataset and delete rows from the old dataset (and save it as well)
        df.to_csv(datasets[i], index=False)
        df_to_extract_from.to_csv(dataset_to_extract_from_p, index=False)

        # Print progress
        if i==0:
            print("TRAIN DATASET LABELLING COMPLETED")
        else:
            print("VALIDATION DATASET LABELLING COMPLETED")