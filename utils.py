import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import re
from functools import reduce
import nltk
from nltk.corpus import stopwords


def load_dataframe():
    """
    Load dataframe from texts in data folder.
    -Text is all the text in each file
    -Label is the name of the folder containin the file
    """
    texts = []
    labels = []
    for folder in os.listdir('data'):
        for file in os.listdir('data/'+folder):
            with open('data/'+folder+'/'+file, 'r') as f:
                texts.append(f.read())
                labels.append(folder)
    df = pd.DataFrame({'text':texts, 'label':labels})
    return df

def plot_word_distribution(df):
    """
    Plot the distribution of word count in each text.
    """
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12,6))
    sns.distplot(df['word_count'])
    plt.title('Distribution of Word Count')
    plt.show()

def lower(text):
    """
    Transforms given text to lower case.
    """
    return text.lower()
 
def replace_special_characters(text):
    """
    Replaces special characters, such as paranthesis, with spacing character
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    return REPLACE_BY_SPACE_RE.sub(' ', text)
 
def replace_br(text):
    """
    Replaces br characters
    """
    return text.replace('\n', ' ')

def filter_out_uncommon_symbols(text):
    """
    Removes any special character that is not in the good symbols list (check regular expression)
    """
    GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    return GOOD_SYMBOLS_RE.sub('', text)
 
def remove_stopwords(text):
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words('english'))
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])
 
def strip_text(text):
    """
    Removes any left or right spacing (including carriage return) from text.
    """
    return text.strip()

def text_prepare(text, filter_methods):
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """
    filter_methods = filter_methods
    return reduce(lambda txt, f: f(txt), filter_methods, text)