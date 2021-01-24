import pandas as pd
import os
import re
import settings
import requests
import time
import geocoder
import preprocessing
import csv
import json

old_datasets_location = r"C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project\Code\IEEE_Dataset"
new_datasets_location = old_datasets_location + r"\refactored_datasets"
merged_dataset_location = old_datasets_location + r'\tweet_retrieval\all_twitter_data_refactored.csv'
old_datasets_location = re.sub(r"\\", "/", old_datasets_location)
new_datasets_location = re.sub(r"\\", "/", new_datasets_location)

dataset_field_names = ['tweet_id', 'username', 'user_followers_count', 'coordinates', 'datetime', 'location', 'raw_text', 'sentiment_score']
location_list = ['United States', 'United Kingdom', 'Australia', 'India', 'Canada', 'Nigeria', 'Kenya']
mask_synonyms = ['mask', 'masks', 'wearamask', 'wear a mask', 'wearmask', 'wear mask', 'face shield', 'faceshield', 'masks4all', 'mask for all', 'masks for all']

saved_tweets = []
merged_file_info = None
start_file = 'october9_october10.csv'

# Creates headers to send
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

# Convert lat-long into country using Google Maps
def get_google_location(location, google_maps_api_key):
    result = geocoder.google(location, key=google_maps_api_key)
    if result is None or result.current_result is None:
        return None
    return result.current_result.country_long

# Convert tweet id to time
def tweet_id_to_time(tweet_id):
    tweet_id = int(tweet_id)
    shifted_id = tweet_id >> 22
    timestamp = shifted_id + 1288834974657
    datetime = time.ctime(timestamp / 1000)
    return datetime

# Get tweet info
def get_tweet(tweet_id, twitter_api_headers):
    df = merged_file_info
    # region old_code
    # Get tween by id
    #response = requests.get(
    #    "https://api.twitter.com/1.1/statuses/lookup.json?tweet_mode=extended&x-account-rate-limit-reset&id=" + str(tweet_id),
    #    headers=twitter_api_headers
    #)
    #if response.status_code == 429:
    #    print("Too many requests for Twitter", response.status_code)
    #    return False
    #elif response.status_code != 200:
    #    return False
    #resp = response.json()

    #if resp == []:
    #    return False

    # Get info about tweet
    #if 'retweeted_status' in resp[0].keys():
    #    full_text = resp[0]['retweeted_status']['full_text']
    #else:
    #    full_text = resp[0]['full_text']
    #return resp[0]['user']['screen_name'], resp[0]['place']['full_name'], full_text
    # endregion
    instance = df.loc[df['tweet_id'] == int(tweet_id)]
    try:
        index = instance.index.tolist()[0]
    except:
        return None
    instance = instance.to_dict()
    return instance['user_screen_name'][index], instance['user_followers_count'][index], instance['coordinates'][index], instance['location'][index], instance['text'][index]

# Save tweets to the csv file
def save_tweets_to_file(new_datasets_location, location_list):
    global saved_tweets, dataset_field_names

    # Get number of rows for each file
    location_dict = dict()
    datasets = dict()
    for location in location_list:
        file = re.sub(' ', '_', location) + '.csv'
        full_dataset_name = new_datasets_location + '/' + file.lower()

        with open(full_dataset_name, 'r', encoding='utf-8') as csv_file:
            length = len(csv_file.readlines())
            if length == 1 or length == 0:
                location_dict[full_dataset_name] = 0
            else:
                location_dict[full_dataset_name] = length

    # Add column names if dataset is empty
    for full_dataset_name, num_of_rows in location_dict.items():
        if num_of_rows == 0:
            with open(full_dataset_name, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dataset_field_names)
                writer.writeheader()

        df = pd.read_csv(full_dataset_name)
        datasets[full_dataset_name] = df

    # Add every tweet from the RAM list
    for tweet in saved_tweets:
        # Initialize dataset name for this tweet
        dataset_name = re.sub(' ', '_', tweet['location']) + '.csv'
        full_dataset_name = new_datasets_location + '/' + dataset_name.lower()

        # Add instance in the dataset (if it is not already there)
        if tweet['tweet_id'] not in map(str, datasets[full_dataset_name].tweet_id.values):
            datasets[full_dataset_name] = datasets[full_dataset_name].append(tweet, ignore_index=True)

    # Save new dataframes to csv
    for full_dataset_name, df in datasets.items():
        df.to_csv(full_dataset_name, index=False)

# Save tweet to the list
def save_tweet(new_datasets_location, tweet_id, user_screen_name, user_followers_count, coordinates, datetime, location, raw_text, sentiment_score, save_in_csv):
    global saved_tweets, dataset_field_names, location_list

    # If the entire csv file is not parsed, just save it in RAM
    saved_tweets.append({
        dataset_field_names[0]: tweet_id,
        dataset_field_names[1]: user_screen_name,
        dataset_field_names[2]: user_followers_count,
        dataset_field_names[3]: coordinates,
        dataset_field_names[4]: datetime,
        dataset_field_names[5]: location,
        dataset_field_names[6]: raw_text,
        dataset_field_names[7]: sentiment_score
    })
    if not save_in_csv:
        return
    else:
        save_tweets_to_file(new_datasets_location, location_list)
        saved_tweets = []

# Extract all information from teh dataset
def extract_dataset_info(old_datasets_location, new_datasets_location, mask_synonyms, twitter_api_headers, google_geocode_api_key):
    start = False
    start_time = time.time()

    all_files = list(os.listdir(old_datasets_location))
    for file_number, filename in enumerate(all_files):
        # Start from the starting file
        if filename == start_file:
            start = True
        if not start or not filename.endswith('.csv'):
            continue

        # Preprocess the dataset
        full_filename = old_datasets_location + "/" + filename
        save_in_csv = True
        with open(full_filename, "r") as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                line = [item.strip() for item in line.split(',')]
                tweet_id, sentiment_score = line if len(line) == 2 else (line[0], 'nan')
                datetime = tweet_id_to_time(tweet_id)
                response = get_tweet(tweet_id, twitter_api_headers)

                # Skip if response is None (maybe tweet has been deleted)
                if response is None:
                    continue

                user_screen_name, user_followers_count, coordinates, raw_location, raw_text = response

                # Skip if the text of the tweet does not contain the words relevant to masks
                for word in raw_text.split():
                    if word in mask_synonyms:
                        break
                    if word[0] == '#' and word[1:] in mask_synonyms:
                        break
                else:
                    continue    # if the loop didn't break, (there is no mask word in text), continue to the next tweet

                # Skip if location is not in the desired locations list
                location = get_google_location(raw_location, google_geocode_api_key)
                if location not in location_list:
                    continue

                # Save tweet (but only after the whole file is processed)
                save_tweet(new_datasets_location, tweet_id, user_screen_name, user_followers_count, coordinates, datetime, location, raw_text, sentiment_score, save_in_csv)
                if save_in_csv == True:
                    print(filename + " will start now :)")
                    save_in_csv = False

                # Print the percentage
                percentage_complete = round(i / len(lines) * 100, 2)
                print(str(percentage_complete) + "% completed! " + str(round(time.time()-start_time, 2)) + " seconds passed!")
        print(str(round(file_number/len(all_files)*100, 2)) + "% OF ALL FILES COMPLETED!")

# Save info from merged file
def get_tweet_info(merged_dataset_location):
    global merged_file_info
    merged_file_info = pd.read_csv(merged_dataset_location)

if __name__=='__main__':
    bearer_token = settings.bearer_token
    google_geocode_api_key = settings.google_maps_api_key

    twitter_api_headers = create_headers(bearer_token)
    get_tweet_info(merged_dataset_location)
    extract_dataset_info(old_datasets_location, new_datasets_location, mask_synonyms, twitter_api_headers, google_geocode_api_key)