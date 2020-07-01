from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import json
import os
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.core.exceptions import *
from .forms import NameForm
import pandas as pd
from .pipeline import terms, cleaning
from rest_framework.response import Response
from rest_framework.views import APIView
from lxml import html
from lxml.html.clean import Cleaner
from .models import TermModel
from .serializers import TermModelSerializer
import base64

class TermView ( APIView):
    
    #voc = set(pd.read_csv(os.path.join(os.path.dirname(__file__), '/pipeline/fisma-voc-filtered.csv'))['ngrams'])
    pandas_dataframe_2 = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'full_dgf_jsons_table2.csv'))
    def check_abbvs(self, dict_1, abbvs):
        for x in dict_1['ngrams']:
            if x in list(abbvs['abbv']):
                dict_1['ngrams'][dict_1['ngrams'].index(x)] = x + ' ∙ ' + abbvs['term'][abbvs['abbv'].index(x)]
            if x in list(abbvs['term']):
                dict_1['ngrams'][dict_1['ngrams'].index(x)] = abbvs['abbv'][abbvs['term'].index(x)] + ' ∙ ' + x
    
    def post(self, request): 
    
        f = request.data # the input is a json with 'content' and 'content_type'
        if f['content_type'] == 'html':
            decoded_content = base64.b64decode(f['content']).decode('utf-8')
            raw_text = cleaning.clean_html(decoded_content)
    
        if f['content_type'] == 'xhtml':
            decoded_content = base64.b64decode(f['content'])
            raw_text = cleaning.clean_html(decoded_content)
        
        if f['content_type'] == 'pdf':
            decoded_content = base64.b64decode(f['content']).decode('utf-8')
            raw_text = [s for s in decoded_content.splitlines() if s]

        raw_text = ' '.join(raw_text)

        dict_1, abbvs = terms.analyzeFile(raw_text)
    
        self.check_abbvs(dict_1, abbvs)
    
        pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1)
        #pandas_dataframe_1 = pandas_dataframe_1[pandas_dataframe_1['ngrams'].isin(voc)]

        t1 = pd.merge(pandas_dataframe_1, self.pandas_dataframe_2, how='left', on=['ngrams'])
        t1.df = t1.df + 1
        idf = 240 / t1.df # 240 is number of docs from which pd2 was created, 10320 for europarl
        t1['idf'] = idf
        t1['tfidf'] = t1.tf * t1.idf
        t1 = t1.fillna(0)
        t1 = t1.sort_values(by='tfidf', ascending=False)
        value_to_return = {}
        for x, y in zip(t1['ngrams'], t1['tfidf']):
            value_to_return.update({x : y})
                
        return HttpResponse(str(value_to_return))