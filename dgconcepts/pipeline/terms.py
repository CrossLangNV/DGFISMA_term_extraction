from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from os.path import isfile, join
from . import Ngrams
from .Ngrams import NgramsFinder 
from threading import Thread
from os import listdir
import argparse
import contractions
from .utils import *
import time
import sys
import re
import time
import os
import spacy
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pandas as pd 
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import math
import string
from collections import Counter

#---------------------
#TERM EXTRACTION
#---------------------
def getTermLists(finder): 

    ng1 = []
    ng2 = []
    ng3 = []
    ng4 = []

    for ng in sorted(finder.ngrams[1]):
        ng1.append(ng)

    for ng in sorted(finder.ngrams[2]):
        ng2.append(ng)
        
    for ng in sorted(finder.ngrams[3]):
        ng3.append(ng)
    
    for ng in sorted(finder.ngrams[4]):
        ng4.append(ng)

    return ng1, ng2, ng3, ng4

def calculate_tf(finder, doc_for_tf_idf, full_term_list, doc_id):
    ##### Getting tf for all terms
    #TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
    #IDF: (Total number of sentences (documents))/(Number of sentences (documents) containing the word)
    doc_for_tf_idf = doc_for_tf_idf.replace('\n', '. ')
    lengths = set()
    for term in full_term_list:
        lengths.add(len(term.split())) #finding the longest ngram
    txt1 = []
    txt1.append(doc_for_tf_idf)
    
    all_ngrams_and_frequencies = {'ngrams':[], 'df': []} # all ngrams per doc (for the main corpus)
    ngrams_and_counts = {} #the number of ngrams for tf calculation
    all_ngrams_and_metrics = {"id_doc" : doc_id, "term_ngram" : [], "count":[], "tf":[]} # table of terms per doc
    terms_and_frequencies = [] #intermediate list of terms and frequency counts
    counts = {} #defining the ngram range
    for x in range(1, max(lengths)):
        counts.update({str(x) + '-grams' : []})
        
    #counting the number of n grams per sentence
    for sentence in doc_for_tf_idf.split('.'):
        try:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation.replace("/", "")))
            for x in range(1, min(len(sentence.split()), max(lengths))):
                vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', stop_words=None, ngram_range = (x,x))
                X0 = vectorizer.fit_transform([sentence])
                features = vectorizer.get_feature_names()
                ngrams_and_frequencies = list(zip(features, np.ravel(X0.sum(axis=0))))
                terms_and_frequencies.append(ngrams_and_frequencies)
                ngram_count = sum(n for _, n in ngrams_and_frequencies)
                if str(x) + '-grams' in counts:
                    counts[str(x) + '-grams'].append(ngram_count)
                else:
                    counts.update({str(x) + '-grams':[ngram_count]})
        except:
            continue
    #vectorizer = CountVectorizer(token_pattern=r”(?u)\b\w+\b”, stop_words=None, ngram_range = (x,x)) <= old pattern
    for x in counts:
        ngrams_and_counts.update({x:sum(counts[x])})
    
    #flattening the list of frequencies per term per sentence
    flat_list = [item for sublist in terms_and_frequencies for item in sublist]
    d = {x:0 for x, _ in flat_list} 
    for name, num in flat_list: d[name] += num 
  
    # summing up the frequencies for duplicate terms, the term frequencies are now for the whole corpus
    terms_and_frequencies = list(map(tuple, d.items()))
    for element in terms_and_frequencies:
        term_length = len(element[0].split())
        if element[1] > 1 and term_length <= max(lengths): #getting rid of all entries that occur only once
            all_ngrams_and_frequencies["ngrams"].append(element[0])
            all_ngrams_and_frequencies["df"].append(element[1])
        if element[0] in full_term_list and element[1] > 1 : #sanity check for the uliege terms
            all_ngrams_and_metrics["term_ngram"].append(element[0])
            all_ngrams_and_metrics["count"].append(element[1])
            all_ngrams_and_metrics["tf"].append(element[1] / ngrams_and_counts[str(term_length)+'-grams'])
    
    dict_1 = all_ngrams_and_metrics
    dict_2 = all_ngrams_and_frequencies
    return dict_1, dict_2
    
def formatAllList(finder, lowerTexts, doc_for_tf_idf, doc_id):    

    ng1, ng2, ng3, ng4 = getTermLists(finder)
    full_term_list = ng1 + ng2 + ng3 + ng4
    dict_1, dict_2 = calculate_tf(finder, doc_for_tf_idf, full_term_list, doc_id)
    return dict_1, dict_2

#---------------------
#MAIN FUNCTION
#---------------------
def analyzeFile(texts, lowerTexts , doc_id):

    finder = NgramsFinder(5)   
    doc_for_tf_idf = finder.feedText(texts)
    dict_1, dict_2 = formatAllList(finder, lowerTexts, doc_for_tf_idf, doc_id)
    return dict_1, dict_2
