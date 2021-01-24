import nltk

s = 'Blue Tuesday for that loverly sunshine out there ðŸ’‹#tuesdayvibes #greatday #staysafe #loveyourselffirst #wearamask #happytimesâ¤ï¸ #blue #sequins #sparkle @Iona on Robert https://t.co/ks8rBDKouE'
tokenizer = nltk.TweetTokenizer()
print(tokenizer.tokenize(text=s))