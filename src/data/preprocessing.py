# -*- coding: utf-8 -*-
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

contractions = { 
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'll": "I will",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"o'clock": "of the clock",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that had",
"that's": "that is",
"there'd": "there would",
"there's": "there is",
"they'd": "they had / they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we had",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when has",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"would've": "would have",
"wouldn't": "would not",
"y'all": "you all",
"y'all'd": "you all would",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"you've": "you have"
}

PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE = PUNCT_TO_REMOVE.replace(",", "")

STOPWORDS = set(stopwords.words('english'))

stemmer = PorterStemmer()

def remove_space(text):
    """custom function to remove the weird spaces"""
    text = text.strip()
    text = text.split()
    return " ".join(text)

def remove_punctuation(text):
    """custom function to remove the punctuations like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_contractions(text):
    """custom function to remove the contractions like shan't and convert them to shall not"""
    for key in contractions.keys():
        text = text.replace(key, contractions[key])
    return text

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def preprocess_text_column(df, column_name):
    """ Runs data preprocessing functions on the text column 
        and stores the preprocessed data into that same column
    """
    #Removing weird spaces
    df[column_name] = df[column_name].apply(remove_space)

    #Lower casing
    df[column_name] = df[column_name].str.lower()

    #Removing of Punctuations
    df[column_name] = df[column_name].apply(remove_punctuation)

    #Remove Contractions
    df[column_name] = df[column_name].apply(remove_contractions)

    #Remove Stopwords
    df[column_name] = df[column_name].apply(remove_stopwords)

    #Stemming
    df[column_name] = df[column_name].apply(stem_words)