import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from functools import reduce
import nltk
from nltk.corpus import stopwords
from collections import OrderedDict
import pickle

def load_dataframe():
    """
    Load dataframe from texts in data folder.
    -Text is all the text in each file
    -Label is the name of the folder containin the file
    """
    path = 'dataframe.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    texts = []
    labels = []
    for folder in os.listdir('data'):
        for file in os.listdir('data/'+folder):
            with open('data/'+folder+'/'+file, 'r') as f:
                texts.append(f.read())
                labels.append(folder)
    df = pd.DataFrame({'text':texts, 'label':labels})
    pickle.dump(df, open(path, 'wb'))
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

def remove_emails(text):
    """
    Removes any email from text.
    """
    return re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
 
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

def build_vocabulary(df):
    """
    Given a dataset, builds the corresponding word vocabulary.

    :param df: dataset from which we want to build the word vocabulary (pandas.DataFrame)
    :return:
      - word vocabulary: vocabulary index to word
      - inverse word vocabulary: word to vocabulary index
      - word listing: set of unique terms that build up the vocabulary
    """
    idx_to_word = OrderedDict()
    word_to_idx = OrderedDict()
    
    curr_idx = 0
    for sentence in tqdm(df.text.values):
        tokens = sentence.split()
        for token in tokens:
            if token not in word_to_idx:
                word_to_idx[token] = curr_idx
                idx_to_word[curr_idx] = token
                curr_idx += 1

    word_listing = list(idx_to_word.values())
    return idx_to_word, word_to_idx, word_listing

def evaluate_vocabulary(idx_to_word, word_to_idx, word_listing, df):
    print("[Vocabulary Evaluation] Size checking...")
    assert len(idx_to_word) == len(word_to_idx)
    assert len(idx_to_word) == len(word_listing)

    print("[Vocabulary Evaluation] Content checking...")
    for i in tqdm(range(0, len(idx_to_word))):
        assert idx_to_word[i] in word_to_idx
        assert word_to_idx[idx_to_word[i]] == i

    print("[Vocabulary Evaluation] Consistency checking...")
    _, _, first_word_listing = build_vocabulary(df)
    _, _, second_word_listing = build_vocabulary(df)
    assert first_word_listing == second_word_listing

    print("[Vocabulary Evaluation] Toy example checking...")
    toy_df = pd.DataFrame.from_dict({
        'text': ["all that glitters is not gold", "all in all i like this assignment"]
    })
    _, _, toy_word_listing = build_vocabulary(toy_df)
    toy_valid_vocabulary = set(' '.join(toy_df.text.values).split())
    assert set(toy_word_listing) == toy_valid_vocabulary

def save_vocab(idx_to_word, word_to_idx, word_listing):
    path = 'vocab/'
    pickle.dump(idx_to_word, open(path+'idx_to_word.pkl', 'wb'))
    pickle.dump(word_to_idx, open(path+'word_to_idx.pkl', 'wb'))
    pickle.dump(word_listing, open(path+'word_listing.pkl', 'wb'))

def load_vocab():
    path = 'vocab/'
    idx_to_word = pickle.load(open(path+'idx_to_word.pkl', 'rb'))
    word_to_idx = pickle.load(open(path+'word_to_idx.pkl', 'rb'))
    word_listing = pickle.load(open(path+'word_listing.pkl', 'rb'))
    return idx_to_word, word_to_idx, word_listing

def split_dataframe(df, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    :param df: dataset to be split (pandas.DataFrame)
    :param test_size: size of the test set (float)
    :return: training and testing sets (pandas.DataFrame)
    """
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int((1-test_size)*len(df))
    return df[:train_size], df[train_size:].reset_index(drop=True)