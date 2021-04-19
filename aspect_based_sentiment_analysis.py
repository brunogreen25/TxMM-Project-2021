import preprocessing
import time
from nltk.corpus import wordnet
import nltk
import stanza
import os
import re
import pandas as pd
import json
import settings

datasets_location = settings.refactored_dataset_location
datasets_location = re.sub(r"\\", "/", datasets_location)

nlp = stanza.Pipeline('en')
mask_synonyms = settings.mask_synonyms_list
mask_synonyms_nlp = nlp(' '.join(mask_synonyms))

# Join features of one instance (where nouns are next to each other)
def join_features(data):
    tagged_sentence = []
    sentence = []

    noun_phrase = ''

    # Join nouns
    for i in range(len(data)):
        if data[i][1][:2]=='NN':
            noun_phrase += data[i][0]
        else:
            if noun_phrase != '':
                sentence.append(noun_phrase)
                tagged_sentence.append((noun_phrase, 'NN'))
                noun_phrase = ''

            sentence.append(data[i][0])
            tagged_sentence.append(data[i])

    if noun_phrase != '':
        sentence.append(noun_phrase)
        tagged_sentence.append((noun_phrase, 'NN'))
        noun_phrase = ''

    return sentence, tagged_sentence

# Get relations between words
def get_relations(sentence, tagged_sentence):
    doc = nlp(' '.join(sentence))

    # Create dependency tree
    dependencies = []
    for dep in doc.sentences[0].dependencies:
        dependencies.append([dep[2].text, dep[0].id, dep[1]])

    sentence = [dep[0] for dep in dependencies]

    try:
        # Change index of the dependency tree with corresponding word
        for i, dep in enumerate(dependencies):
            if dep[1] != 0:
                dependencies[i][1] = sentence[dep[1] - 1]
    except:
        print("Error")

    return dependencies

# Filter only desired POS taggs
def filter_pos(sentence, tagged_list):
    new_sentence = []
    new_tagged_list = []

    desired_pos = ['NN', 'JJ', 'RB']

    for word, pos_tag in tagged_list:
        if pos_tag[:2] in desired_pos:
            new_tagged_list.append((word, pos_tag))
            new_sentence.append(word)

    return new_sentence, new_tagged_list

# Get cluster from the sentence
def get_cluster(tagged_sentence, relations):
    cluster = []
    desired_deps = ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"]

    for word, pos_tag in tagged_sentence:
        descript_words = []
        for rel in relations:
            if rel[0] == word and rel[2] in desired_deps and pos_tag[:2] == 'NN':
                descript_words.append(rel[1])
            if rel[1] == word and rel[2] in desired_deps and pos_tag[:2] == 'NN':
                descript_words.append(rel[0])
        # Add words if description_list is not empty and the word is of interest
        if descript_words != []:        # and any([w in word for w in mask_synonyms])
            for mask_word in mask_synonyms:
                if mask_word in word:
                    cluster.append((mask_word, descript_words))
                    break
    return cluster

# Add negations where needed
def add_negations_to_cluster(cluster, negations):
    for neg, word in negations:
        for i in range(len(cluster)):
            for j in range(len(cluster[i][1])):
                if word == cluster[i][1][j]:
                    cluster[i][1][j] = neg + ' ' + cluster[i][1][j]

    return cluster

# Extract negations and the word it is refered to
def extract_negations(sentence):
    negations = []
    neg_list = ['not', 'no']

    for i, word in enumerate(sentence):
        if word in neg_list and i != len(sentence)-1:
            negations.append((sentence[i], sentence[i+1]))

    return negations

def get_aspect(data):
    sentence, tagged_sentence = preprocessing.preprocess(data, pos_tagging=True, remove_stopwords=False)

    sentence, tagged_sentence = join_features(tagged_sentence)
    negations = extract_negations(sentence)

    relations = get_relations(sentence, tagged_sentence)
    sentence, tagged_sentence = filter_pos(sentence, tagged_sentence)

    cluster = get_cluster(tagged_sentence, relations)
    cluster = add_negations_to_cluster(cluster, negations)

    return cluster

if __name__ == '__main__':

    # Iterate all datasets
    files = os.listdir(datasets_location)
    for file_num, filename in enumerate(files):
        full_filename = datasets_location + '/' + filename
        new_file_path = datasets_location + '/new_' + filename
        df = pd.read_csv(full_filename)
        new_column = {
            'aspect_terms': []
        }
        n_rows = df.shape[0]

        # Extract aspects
        for index, row in df.iterrows():
            aspect = get_aspect(row['raw_text'])

            # Convert aspect to JSON and replace comma, so it is safer for csv
            json_aspect = json.dumps(aspect)
            json_aspect = re.sub(',', ';', json_aspect)

            new_column['aspect_terms'].append(aspect)

            # Print progress
            if index % 30 == 0:
                print(file_num+1, str(round(index/n_rows*100,2)) + '% complete')

        # Save to the dataframe
        df['aspects'] = new_column['aspect_terms']
        df.to_csv(new_file_path, index=False)
        print(filename + " saved")