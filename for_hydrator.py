import os
import re
import pandas as pd

### MERGE ALL FILES
old_datasets_location = r"C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset"
new_datasets_location = old_datasets_location + r"\refactored_datasets"
old_datasets_location = re.sub(r"\\", "/", old_datasets_location)
new_datasets_location = re.sub(r"\\", "/", new_datasets_location)
tweet_ids = []

#for i, filename in enumerate(os.listdir('IEEE_Dataset')):
#    if not filename.endswith('.csv'):
#        continue
#    full_filename = old_datasets_location + '/' + filename
#    with open(full_filename, 'r') as fp:
#        for line in fp.readlines():
#            tweet_ids.append(line.split(',')[0].strip())

#print(tweet_ids[245548:245550])
#with open(old_datasets_location + '/tweet_retrieval/all_ids.txt', 'w') as fp:
#    for tweet_id in tweet_ids:
#        fp.write(tweet_id+'\n')

## SELECT ONLY NEEDED COLUMNS
df = pd.read_csv(old_datasets_location+'/tweet_retrieval/all_twitter_data.csv')
df_refactored = df[['id', 'user_screen_name', 'user_followers_count', 'coordinates', 'place', 'text']]
df_refactored.to_csv(old_datasets_location+'/tweet_retrieval/all_twitter_data_refactored.csv', index=False)