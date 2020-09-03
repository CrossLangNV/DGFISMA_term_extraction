from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from os.path import isfile, join
from . import Ngrams
from .Ngrams import NgramsFinder
from threading import Thread
from os import listdir
import argparse
import contractions
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
#ABBREVIATION EXTRACTION
#---------------------
def extractAbbvTerm(tokens,i):
    """
    Task
    ----
        extracts extended form of an abbreviation
    Args
    ----
        tokens,
            tokenized text to analyze
            
        i,
            index of abbreviation token
            
    Output
    ------
        extended form if any else None
    """  
    abbv = tokens[i]
    k = 1
    for j,c in enumerate(abbv[::-1]):
        sw = set(stopwords.words('english')) 
        while(i-j-k >= 0 and tokens[i-j-k][0] != c and c.isupper() and (tokens[i-j-k] in sw or tokens[i-j-k] == '’' or tokens[i-j-k][0] == '(')):
            k += 1
        if(i-j-k < 0 or (c.isupper() and tokens[i-j-k][0].lower() != c.lower())):
            return None
    
    res = ""
    for t in tokens[i-len(abbv)-k+1:i]:
        if(t != "’" and t != "(" and t != "{" and t != "["):
            res +=t+" "    
    return res

def extractAbbv(tokens):
    """
    Task
    ----
        Extract all token which are possible candidate for abbreviation
    Args
    ----
        tokens,
            Tokens to to analyze
            
    Output
    ------
        list of candidate abbreviation
    """  
    sw = set(stopwords.words('english')) 
    res = []    
    for i,t in enumerate(tokens):
        prop = sum(1 for c in t if c.isupper())/len(t)
        if(prop > 0.5 
                and len(t) < 6 
                and len(t) > 1 
                and t.lower() not in sw 
                and sum(1 for c in t if c == 'V' or c=='I') != len(t) 
                and t.isalpha()):
            term = extractAbbvTerm(tokens,i)
            if(term is not None):
                res.append((t,term))
    return list(set(res))

def calculate_tf(doc_for_tf_idf, full_term_list):
    lengths = set()
    for term in full_term_list:
        lengths.add(len(term.split()))  # finding the longest ngram
    txt1 = [doc_for_tf_idf]

    ngrams_and_counts = {}  # the number of ngrams for tf calculation
    all_ngrams_and_metrics = {"ngrams" : [], "count":[], "tf":[]} # table of terms per doc
    terms_and_frequencies = []  # intermediate list of terms and frequency counts
    counts = {}  # defining the ngram range
    for x in range(1, max(lengths)):
        counts.update({str(x) + '-grams' : []})
        
    for sentence in doc_for_tf_idf.split('.'):
        try:
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
        except ValueError:
            pass

    for x in counts:
        ngrams_and_counts.update({x:sum(counts[x])})
    
    # flattening the list of frequencies per term per sentence
    flat_list = [item for sublist in terms_and_frequencies for item in sublist]
    d = {x:0 for x, _ in flat_list} 
    for name, num in flat_list: d[name] += num 
  
    # summing up the frequencies for duplicate terms, the term frequencies are now for the whole corpus
    terms_and_frequencies = list(map(tuple, d.items()))
    for element in terms_and_frequencies:
        term_length = len(element[0].split())
        if element[0] in full_term_list:  # terms that only occur once are also included here
            all_ngrams_and_metrics["ngrams"].append(element[0])
            all_ngrams_and_metrics["count"].append(element[1])
            all_ngrams_and_metrics["tf"].append(element[1] / ngrams_and_counts[str(term_length)+'-grams'])

    return all_ngrams_and_metrics

#---------------------
#MAIN FUNCTION
#---------------------
def analyzeFile(texts):
    # TO DO : annotations of abbreviations
    finder = NgramsFinder(4)
    doc_for_tf_idf, ngrams = finder.feedText(texts)
    all_ngrams = []
    for x in ngrams:
        all_ngrams.append([term for term in x])
    terms_v1 = [term for ngrams in all_ngrams for term in ngrams]
    abbv = extractAbbv(word_tokenize(texts))
    abvs = []
    for x in abbv:
        abv = x[0] + " ■ " + x[1]
        terms_v1.append(x[0].lower().strip())
        terms_v1.append(x[1].lower().strip())
        abvs.append(abv)
    dict_1 = calculate_tf(doc_for_tf_idf, terms_v1)
    return dict_1, abvs