from django.shortcuts import render
from django.http import HttpResponse
import json
import os
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.core.exceptions import *
from .forms import NameForm
import pandas as pd
import contractions
from .pipeline import terms
from rest_framework.response import Response
from rest_framework.views import APIView
from lxml import html
from lxml.html.clean import Cleaner
from .models import TermModel
from .serializers import TermModelSerializer
import base64

class TermView ( APIView):
    
    #for a single text doc
    def post(self, request): # here we should also add pd_2 as input
        doc_id = 1 # should be unique for each input, to be amended
        f = request.data
        decoded_content = base64.b64decode(f['content']).decode('utf-8')
        decoded_content = Cleaner(style=True, links=True, add_nofollow=True, page_structure=False, safe_attrs_only=False).clean_html(decoded_content)
        raw_text = html.fromstring(decoded_content)
        raw_text = [s for s in raw_text.text_content().splitlines() if s]
        lowerTexts = []
        for x in raw_text:
            lowerTexts.append(x.lower())
        f = '. '.join(raw_text)
        dict_1, dict_2 = terms.analyzeFile(f,lowerTexts, doc_id)
        list_of_terms = []
        for x in list(zip(dict_1['term_ngram'], dict_1['count'])):
            list_of_terms.append(x)
        #pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1) #(id_doc | term x-gram | count x-gram | tf )
        #pandas_dataframe_2 = pd.DataFrame.from_dict(dict_2) # literally all ngrams, not just terms
        return HttpResponse(str(list_of_terms))
    
    """
    def post(self, request):
    #for multiple text docs
        corpus_table = pd.DataFrame(columns = ["ngrams", "df"])
        doc_id = 0
        list_of_terms = {}
        texts = {}    
        rqp = request.data['rqp']
        #rqp = '/path/to/dir/with/texts/'
        for fileName in os.listdir(rqp):
            print(fileName)
            doc_id +=1
            f = open(os.path.join( rqp, fileName))
            texts[fileName] = f.read()
            lowerTexts = texts[fileName].lower()
            lowerTexts = os.linesep.join([s for s in lowerTexts.splitlines() if s]) # removing possible empty lines
            dict_1, dict_2 = terms.analyzeFile(fileName,texts,lowerTexts, doc_id)
            pandas_dataframe_2 = pd.DataFrame.from_dict(dict_2) # literally all ngrams, not just terms
            corpus_table = corpus_table.append(pandas_dataframe_2)
            list_of_terms[fileName] = list(zip(dict_1['term_ngram'], dict_1['count']))
        
        aggregation_functions = {'df': 'sum'}
        corpus_table = corpus_table.groupby(corpus_table['ngrams']).aggregate(aggregation_functions)
        corpus_table.to_csv('/Users/alina/Desktop/test.csv')
        return HttpResponse(str(list_of_terms))
    """
