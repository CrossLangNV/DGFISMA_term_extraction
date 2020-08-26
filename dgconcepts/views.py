import base64
import os
import re
import time
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from rest_framework.views import APIView
from .pipeline import terms, cleaning, annotations
from annotations import add_terms_to_cas
from cleaning import get_text_html
import cassis
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi


class TermView(APIView):
    dic = set(line.strip() for line in
              open(os.path.join(settings.MEDIA_ROOT, 'words.txt')))  # voc file for pdf preprocessing
    pandas_dataframe_2 = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'full_dgf_jsons_table2.csv'))

    def check_abbvs(self, dict_1, abbvs):
        for x in dict_1['ngrams']:
            if x in list(abbvs['abbv']):
                dict_1['ngrams'][dict_1['ngrams'].index(x)] = x + "■" + abbvs['term'][abbvs['abbv'].index(x)]
            if x in list(abbvs['term']):
                dict_1['ngrams'][dict_1['ngrams'].index(x)] = abbvs['abbv'][abbvs['term'].index(x)] + "■" + x

    def update_voc(self, dict_1):
        voc = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'fisma-voc-filtered.csv'))
        raw_terms_and_counts = list(zip(dict_1['ngrams'], dict_1['count']))
        raw_terms_and_counts = pd.DataFrame(raw_terms_and_counts, columns=['ngrams', 'count']).sort_values(by='count',
                                                                                                           ascending=False)
        aggregation_functions = {'count': 'sum'}
        voc = voc.append(raw_terms_and_counts)
        new_voc = voc.groupby(voc['ngrams']).aggregate(aggregation_functions)
        new_voc = new_voc.sort_values(by='df', ascending=False)
        new_voc.to_csv(os.path.join(settings.MEDIA_ROOT, 'fisma-voc-filtered.csv'))

    def update_pd2(self, dict_1):
        pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1)
        corpus_table = self.pandas_dataframe_2.append(pandas_dataframe_1)
        aggregation_functions = {'df': 'sum'}
        corpus_table = corpus_table.groupby(corpus_table['ngrams']).aggregate(aggregation_functions)
        corpus_table = corpus_table.sort_values(by='df', ascending=False)
        corpus_table.to_csv(os.path.join(settings.MEDIA_ROOT, 'full_dgf_jsons_table2.csv'))

    def post(self, request):
        with open(os.path.join(settings.MEDIA_ROOT, 'rb')) as f:
            typesystem = load_typesystem(f)

        start = time.time()
        f = request.data  # the input is a json with 'content' and 'content_type'
        cas = load_cas_from_xmi(f['xmi'], typesystem=typesystem)
        sofa_id = "html2textView"
        sentences = get_text_html(cas, sofa_id, tagnames=['p'])

        if f['content_type'] == 'html':
            decoded_content = base64.b64decode(f['content']).decode('utf-8')
            raw_text = cleaning.clean_html(decoded_content)

        if f['content_type'] == 'xhtml':
            decoded_content = base64.b64decode(f['content'])
            raw_text = cleaning.clean_html(decoded_content)

        if f['content_type'] == 'pdf':
            decoded_content = base64.b64decode(f['content']).decode("utf-8").replace('\xad', '')
            newlines = re.findall(r'[a-zA-z]+\n[a-zA-z]+', str(decoded_content))
            for x in newlines:
                if not x.split('\n')[0].lower() in self.dic and not x.split('\n')[1].lower() in self.dic and str(
                        x.split('\n')[0].lower() + x.split('\n')[1]).lower() in self.dic:
                    decoded_content = decoded_content.replace(x, x.replace('\n', ''))
            otherlines = re.findall(r'[a-zA-z]+\-\n[a-zA-z]+', str(decoded_content))
            for x in otherlines:
                if not x.split('\n')[0].lower() in self.dic and not x.split('\n')[1].lower() in self.dic and str(
                        x.split('\n')[0].lower() + x.split('\n')[1]).lower() in self.dic:
                    decoded_content = decoded_content.replace(x, x.replace('\n', ''))
            raw_text = [s for s in decoded_content.splitlines() if s]

        raw_text = ' '.join(raw_text)

        dict_1, abbvs = terms.analyzeFile(raw_text)

        if f['update_voc']:
            self.update_voc(dict_1)

        if f['update_pd2']:
            self.update_pd2(dict_1)

        self.check_abbvs(dict_1, abbvs)

        pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1)
        # pandas_dataframe_1 = pandas_dataframe_1[pandas_dataframe_1['ngrams'].isin(voc)]

        t1 = pd.merge(pandas_dataframe_1, self.pandas_dataframe_2, how='left', on=['ngrams'])
        t1.df = t1.df + 1
        idf = 240 / t1.df  # 240 is number of docs from which pd2 was created, 10320 for europarl
        t1['idf'] = idf
        t1['tfidf'] = t1.tf * t1.idf
        t1 = t1.fillna(0)
        t1 = t1.sort_values(by='tfidf', ascending=False)
        value_to_return = {}
        for x, y in zip(t1['ngrams'], t1['tfidf']):
            value_to_return.update({x: y})
        cas = add_terms_to_cas(cas, typesystem, sofa_id, [(k, v) for k, v in value_to_return.items()])
        cas_string = cas.to_xmi()
        end = time.time()
        print(end - start)
        return HttpResponse({'terms': cas_string})
        # return HttpResponse(str(value_to_return))
