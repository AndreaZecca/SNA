import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import re
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import pickle

# check if ntlk punkt is installed and download if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def plot_label_distribution(dfs, labels):
    """
    Plot the distribution of word count in each text.
    """
    if len(dfs) != len(labels):
        raise Exception("Number of dataframes and labels must be equal.")
    plt.figure(figsize=(12,6))
    i = 1
    for df, subtitle in zip(dfs, labels):
        plt.subplot(1, len(dfs), i)
        plt.title(subtitle)
        label_count = df['label'].value_counts()
        # sort label_count by name
        label_count = label_count.sort_index()
        # plot the label count without the label description
        label_count.plot(kind='bar')
        i += 1
    # plt.legend()
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

def compute_pmi(text):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(word_tokenize(text))
    pmi = finder.score_ngrams(bigram_measures.pmi)
    tokens = list(set(word_tokenize(text)))
    n = len(tokens)
    ppmi = np.zeros((n,n))
    word_to_idx = {}
    idx_to_word = {}
    for i, token in enumerate(tokens):
        word_to_idx[token] = i
        idx_to_word[i] = token
    for bigram, score in pmi:
        first_word = bigram[0]
        second_word = bigram[1]
        first_idx = word_to_idx[first_word]
        second_idx = word_to_idx[second_word]
        ppmi[first_idx, second_idx] = max(0, math.ceil(score*100)/100)
    return ppmi, word_to_idx, idx_to_word

def plot_pmi(pmi, idx_to_word):
    plt.figure(figsize=(20,10))
    sns.heatmap(pmi, xticklabels=idx_to_word.values(), yticklabels=idx_to_word.values(), cmap='Blues')
    plt.show()


def get_nodes_weights(pmi, idx_to_word):
    weights = {}
    for i in range(len(pmi)):
        weights[idx_to_word[i]] = sum(pmi[i,:])
    return weights

def plot_pmi_graph(pmi, idx_to_word, pmi_word_to_idx):
    G = nx.Graph()
    weights = get_nodes_weights(pmi, idx_to_word)
    for i in pmi_word_to_idx:
        if weights[i] > 0:
            G.add_node(i, weight=weights[i])
    for i in G.nodes():
        for j in G.nodes():
            i_idx = pmi_word_to_idx[i]
            j_idx = pmi_word_to_idx[j]
            if pmi[i_idx, j_idx] > 0:
                G.add_edge(i, j, weight=pmi[i_idx, j_idx])
    plt.figure(figsize=(30,15))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[i]['weight']*100 for i in G.nodes()])
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()
    
def threshold_pmi(pmi, threshold):
    pmi_copy = pmi.copy()
    pmi_copy[pmi_copy < threshold] = 0
    return pmi_copy