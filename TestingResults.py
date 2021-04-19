import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
dir_path = r'C:\Users\sluzb\Documents\Fakultet\Radboud\Semester 1\Text And Multimedia Mining\Project_2\IEEE_Dataset\refactored_datasets\v2'
dir_path = re.sub(r'\\', '/', dir_path)

# Number of sentiment scores per country (seperate bar plots)
def code1():
    sentiments = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue

        df = pd.read_csv(dir_path + '/' + filename)
        country = filename[4:][:-13]
        sentiments[country] = np.array([float(sentiment[1:-1]) for sentiment in df.model_output])
        sentiments[country] = np.around((sentiments[country] + 1) * 5 / 2, decimals=0)
        plt.figure(i)
        plt.title(country)
        opinions = np.array([sentiments[country].tolist().count(i) for i in np.arange(0, 6, 1)])
        opinions_percentage = opinions / sum(opinions) * 100
        plt.xlabel('Sentiment Score')
        plt.ylabel('Percentage of Tweets')
        plt.bar(np.arange(0, 6, 1).tolist(), opinions_percentage)
        plt.show()

# Most frequent words
def code2():
    mask_synonyms = ['mask', 'masks', 'wearamask', 'wear a mask', 'wearmask', 'wear mask', 'face shield',
                     'faceshield', 'masks4all', 'mask for all', 'masks for all']
    stopwords_ = set(stopwords.words('english'))
    def filter_pos(sentence, tagged_list):
        new_sentence = []
        new_tagged_list = []

        desired_pos = ['NN', 'JJ', 'RB']

        i = 0
        for word, pos_tag in tagged_list:
            if pos_tag[:2] in desired_pos and word not in mask_synonyms:
                word_ = word if i == 0 or sentence[i - 1] != 'not' else 'not-' + word

                new_tagged_list.append((word_, pos_tag))
                new_sentence.append(word_)

            i += 1

        return new_sentence, new_tagged_list
    def get_words(text):
        # Filter only desired POS taggs
        sentence, tagged_sentence = preprocessing.preprocess(text, pos_tagging=True, remove_stopwords=True)
        sentence, tagged_sentence = filter_pos(sentence, tagged_sentence)
        return sentence
    def get_frequency(words):
        word_dict = {}
        for word in set(words):
            word_dict[word] = words.count(word)
        word_dict = {k:v for k,v in sorted(word_dict.items(), reverse=True, key=lambda item:item[1])}
        return word_dict
    def parse_aspect(aspect):
        asterix = []
        words = []
        word = ''
        wording = False
        for char in aspect:
            if wording == True and char != "'":
                word += char
            elif char == "'" and asterix != []:
                asterix.pop()
            elif char == "'" and wording == False and asterix == []:
                wording = True
            elif char == "'" and wording == True and asterix == []:
                wording = False
                words.append(word)
                word = ''
            elif char == '(':
                asterix.append("'")
                asterix.append("'")
        return words
    def filter_aspects(aspect):
        filtered_aspects = []
        for aspect in aspects:
            if aspect not in mask_synonyms and aspect not in stopwords_:
                filtered_aspects.append(aspect)
        return filtered_aspects


    positive_words = {}
    negative_words = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue

        df = pd.read_csv(dir_path + '/' + filename)

        country = filename[4:][:-13]
        positive_words[country] = []
        negative_words[country] = []

        for index, row in df.iterrows():
            sent_score = row.model_output
            sent_score = round((float(sent_score[1:-1]) + 1) * 5 / 2, 1)
            if sent_score >= 2.5:
                #positive_words[country]+=get_words(row.raw_text)
                aspects = parse_aspect(row.aspects)
                filtered_aspects = filter_aspects(aspects)
                positive_words[country]+=filtered_aspects
            elif sent_score <= 2.5:
                aspects = parse_aspect(row.aspects)
                filtered_aspects = filter_aspects(aspects)
                #negative_words[country]+=get_words(row.raw_text)
                negative_words[country]+=filtered_aspects
        print(filename)

        positive_words[country] = get_frequency(positive_words[country])
        negative_words[country] = get_frequency(negative_words[country])
        n1 = 0
        n2 = 10

        most_pop_words_keys_pos = list(positive_words[country].keys())[n1:n2]
        most_pop_words_vals_pos = list(positive_words[country].values())[n1:n2]
        most_pop_words_keys_neg = list(negative_words[country].keys())[n1:n2]
        most_pop_words_vals_neg = list(negative_words[country].values())[n1:n2]

        #plt.figure(i, figsize=(12,5))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
        country_names = country.split('_')
        country_name = ' '.join([name[0].upper() + name[1:] for name in country_names])

        ax1.set_title(country_name + ' positive words')
        ax1.set_xlabel("Words")
        ax1.set_ylabel("Number of occurences")
        ax1.bar(most_pop_words_keys_pos, most_pop_words_vals_pos)

        ax2.set_title(country_name + ' negative words')
        ax2.set_xlabel("Words")
        ax2.set_ylabel("Number of occurences")
        ax2.bar(most_pop_words_keys_neg, most_pop_words_vals_neg)
        plt.show()

# Number of tweets with #wearamask
def code3():
    wearamask = {}
    total = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue
        df = pd.read_csv(dir_path + '/' + filename)

        country = filename[4:][:-13]
        wearamask[country] = 0
        total[country] = 0

        for index, row in df.iterrows():
            if 'wearamask' in row.raw_text:
            #if 'hero' in row.raw_text:
                wearamask[country] += 1
            total[country] += 1

        wearamask[country] /= total[country]
        wearamask[country] *= 100

    plt.figure(i)
    plt.title('Tweets with #wearamask')
    plt.xlabel("Country Code")
    plt.ylabel("Percentage of total tweets")
    abbreviations_of_countries = ['AU', 'CA', 'IN', 'NG', 'UK', 'US']
    plt.bar(abbreviations_of_countries, list(wearamask.values()))
    plt.show()

# Correlation of model_output and number of followers
def code4():
    scores = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue
        df = pd.read_csv(dir_path + '/' + filename)

        country = filename[4:][:-13]
        scores[country] = {}

        for index, row in df.iterrows():
            n_fol = row.user_followers_count
            sent = round((float(row.model_output[1:-1]) + 1) * 5 / 2)
            scores[country][sent] = scores[country][sent]+n_fol if sent in scores[country].keys() else n_fol

        for i in range(1,6):
            if i not in scores[country]:
                scores[country][i] = 0

        plt.figure(i, figsize=(12,10))
        country_names = country.split('_')
        country_name = ' '.join([name[0].upper() + name[1:] for name in country_names])
        plt.title('Correlation between number of followers and sentiment scores of users in ' + country_name)
        plt.xlabel("Sentiment score")
        plt.ylabel("Total number of followers")
        followers = [scores[country][i] for i in range(1,6)]
        d = {i+1: f for i, f in enumerate(followers)}
        print(d)
        print(f"WEIGHTED SENT.SCORE for {country_name}:")
        print(sum([k*v for k,v in d.items()])/sum([v for k,v in d.items()]))
        plt.xticks(np.arange(1,6,step=1))
        plt.bar(np.arange(1,6,1), followers)
        plt.show()

# What is the most popular #
def code5():
    def get_hashes(text):
        hashes = []
        sentence = text.split(' ')
        for word in sentence:
            if len(word) >0 and word[0] == '#':
                hashes.append(word)
        return hashes

    most_pop_hash = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue

        df = pd.read_csv(dir_path + '/' + filename)

        country = filename[4:][:-13]
        most_pop_hash[country] = {}

        for index, row in df.iterrows():
            hashes = get_hashes(row.raw_text)
            for hash in hashes:
                if hash in most_pop_hash[country].keys():
                    most_pop_hash[country][hash] += 1
                else:
                    most_pop_hash[country][hash] = 0
        most_pop_hash[country] = {k: v for k, v in sorted(most_pop_hash[country].items(), reverse=True, key=lambda item: item[1])}

        # PLOT MOST FAMOUS TAGS FOR EACH COUNTRY
        plt.figure(i)
        plt.title('Tweets with #wearamask')
        plt.xlabel("Country Code")
        plt.ylabel("Percentage of total tweets")
        abbreviations_of_countries = ['AU', 'CA', 'IN', 'KE', 'NG', 'UK', 'US']
        #plt.bar(abbreviations_of_countries, list(most_pop_hash[country].values()))
        #plt.show()

    # Count the most frequent word in all countries
    overall_winner = dict()
    for _, word_dict in most_pop_hash.items():
        for word, freq in word_dict.items():
            if word in overall_winner.keys():
                overall_winner[word] += int(freq)
            else:
                overall_winner[word] = int(freq)
    overall_winner = {k: v for k, v in sorted(overall_winner.items(), reverse=True, key=lambda item: item[1])}
    print(overall_winner)

# Count number of different users
def code6():
    users = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue

        df = pd.read_csv(dir_path + '/' + filename)

        country = filename[4:][:-13]
        users[country] = set()

        for index, row in df.iterrows():
            username = row.username
            users[country].add(username)

        users[country] = len(users[country])

    print(users)

# Calculate mean and variance
def code7():
    sentiments = {}

    for i, filename in enumerate(os.listdir(dir_path)):
        if not filename.endswith('5_v2.csv'):
            continue

        df = pd.read_csv(dir_path + '/' + filename)
        country = filename[4:][:-13]
        sentiments[country] = np.array([float(sentiment[1:-1]) for sentiment in df.model_output])
        sentiments[country] = np.around((sentiments[country] + 1) * 5 / 2, decimals=0)
        mean = sentiments[country].mean()
        variance = sentiments[country].var()

        print(country, mean, variance)

if __name__=='__main__':
    code4()