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


global nlp
nlp = spacy.load("en_core_web_lg") 
global spacy_stopwords
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
global labels
labels = ['ADP', 'VERB', 'PRON', 'CCONJ', 'SCONJ' ]
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

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)
    
        
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

def calculate_tf(finder, doc_for_tf_idf, full_term_list):
    ##### Getting tf for all terms
    #TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
    #IDF: (Total number of sentences (documents))/(Number of sentences (documents) containing the word)
    doc_for_tf_idf = doc_for_tf_idf.replace('\n', '. ')
    lengths = set()
    for term in full_term_list:
        lengths.add(len(term.split())) #finding the longest ngram
    txt1 = []
    txt1.append(doc_for_tf_idf)
    
    ngrams_and_counts = {} #the number of ngrams for tf calculation
    all_ngrams_and_metrics = {"ngrams" : [], "count":[], "tf":[]} # table of terms per doc
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
        except ValueError:
            pass
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

        if element[0] in full_term_list: #terms that only occur once are included here
        
            all_ngrams_and_metrics["ngrams"].append(element[0])
            all_ngrams_and_metrics["count"].append(element[1])
            all_ngrams_and_metrics["tf"].append(element[1] / ngrams_and_counts[str(term_length)+'-grams'])
    
    dict_1 = all_ngrams_and_metrics

    return dict_1


def formatAllList(finder, doc_for_tf_idf):    

    ng1, ng2, ng3, ng4 = getTermLists(finder)
    full_term_list = ng1 + ng2 + ng3 + ng4
    dict_1 = calculate_tf(finder, doc_for_tf_idf, full_term_list)

    return dict_1

#---------------------
#MAIN FUNCTION
#---------------------
def analyzeFile(texts):

    finder = NgramsFinder(5)   
    doc_for_tf_idf = finder.feedText(texts)
    abbv = extractAbbv(word_tokenize(texts))
    abbv = pd.DataFrame(abbv, columns =['abbv', 'term'])
    dict_1 = formatAllList(finder, doc_for_tf_idf)
    return dict_1, abbv