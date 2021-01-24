import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.corpus import words

nltk.download('stopwords')
nltk.download('words')

vocab = set(words.words())
mask_synonyms = ['mask', 'masks', 'wearamask', 'wear a mask', 'wearmask', 'wear mask', 'face shield', 'faceshield', 'masks4all', 'mask for all', 'masks for all']

emoticons = {
    ':(': 'sad',
    ':-(': 'sad',
    ':)': 'happy',
    ':D': 'very happy',
    ':-)': 'happy',
    ':O': 'surprised',
    ':\'(': 'crying',
    ':\"(': 'crying'
}

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess(tweet: str, pos_tagging=True, remove_stopwords=True, remove_numbers= True, join_text=False):
    global stopwords, emoticons, vocab, mask_synonyms
    original_text = tweet

    # Fix ' error
    tweet = re.sub(r'â€™', "'", tweet)

    # Turn letters to lowercase
    tweet = tweet.lower()

    # Replace contractions
    tweet = tweet.replace("’", "'")
    tweet = decontracted(tweet)

    # Tokenization
    tokenizer = TweetTokenizer()
    tweet_tokens_ = tokenizer.tokenize(tweet)

    # Replace emoticons
    tweet_tokens = list()
    for tweet_token in tweet_tokens_:
        tweet_tokens.append(emoticons[tweet_token] if tweet_token in emoticons.keys() else tweet_token)
    tweet = ' '.join(tweet_tokens)

    # Replace emojis
    tweet = emoji.demojize(tweet, delimiters=(' ', ' '))
    tweet = ' '.join(tweet.split())  # This may lead to double spaces, so concatenate again

    # Remove mentions (@)
    tweet = re.sub(r'''@\w+ ''', '', tweet)

    # Remove HTML tags
    tweet = BeautifulSoup(tweet, features='html.parser').get_text()

    # Tokenization
    tokenizer = TweetTokenizer()
    tweet_tokens_ = tokenizer.tokenize(tweet)

    # Remove URLs
    tweet_tokens = list()
    for tweet_token in tweet_tokens_:
        if not tweet_token.startswith('http') and not tweet_token.startswith('www'):
            tweet_tokens.append(tweet_token)

    # Remove punctuation
    tweet = ' '.join(tweet_tokens)
    punct = '''!()-—[]{};:'"\, <>./?@#$%^&*_~µω⁰…'''
    for sign in punct:
        tweet = tweet.replace(sign, ' ')
    tweet = ' '.join(tweet.split())

    # Tokenization (and only leave words present in vocab)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_tokens = [token for token in tweet_tokens if token in vocab or any([mask_word in token for mask_word in mask_synonyms])]

    # Remove stopwords
    if remove_stopwords:

        stopwords_ = set(stopwords.words('english'))
        tweet_tokens = [token for token in tweet_tokens if token not in stopwords_]

    # Turn short negations to full words
    try:
        for i in range(len(tweet_tokens)):
            if len(tweet_tokens[i]) < 4:
                continue
            if tweet_tokens[i][-1] == 't' and tweet_tokens[i][-2] == "'" and tweet_tokens[i][-3] == 'n':
                tweet_tokens[i] = tweet_tokens[i][:-2] + ' not'
            if tweet_tokens[i][-1] == 't' and tweet_tokens[i][-2] == "n":
                tweet_tokens[i] = tweet_tokens[i][:-1] + ' not'
        tweet_tokens = ' '.join(tweet_tokens).split(' ')
    except:
        print("H")

    # POS Tagging
    if pos_tagging:
        tweet_tokens = nltk.pos_tag(tweet_tokens)
        sentence = [token[0] for token in tweet_tokens]

        if join_text:
            sentence = ' '.join(sentence)
        return sentence, tweet_tokens
    else:

        if join_text:
            tweet_tokens = ' '.join(tweet_tokens)
        return tweet_tokens
