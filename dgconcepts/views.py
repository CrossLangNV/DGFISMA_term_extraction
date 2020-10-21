import base64
import binascii
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView
import spacy
from .pipeline.metrics import *
from .pipeline.terms import extract_concepts
from .pipeline.annotations import *
import os
from .pipeline.utils import *
import logging

NLP = spacy.load('en_core_web_lg')
WHITELIST = open(os.path.join(settings.MEDIA_ROOT, 'whitelist.txt')).read().splitlines()
BLACKLIST = open(os.path.join(settings.MEDIA_ROOT, 'blacklist.txt')).read().splitlines()
MAX_LEN_NGRAM = 4
with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
    TYPESYSTEM = load_typesystem(f)

class TermView(APIView):

    @staticmethod
    def launch_term_extraction(sentences, f):
        start = time.time()
        all_terms = []
        all_abvs = []
        doc_for_tf_idf = []
        for sentence in sentences:
            doc_for_tf_idf.append(sentence)
            terms_so_far = []
            ngrams, supergrams, abvs = extract_concepts(sentence, NLP, MAX_LEN_NGRAM)
            all_abvs.append(abvs)
            terms_so_far.append(ngrams)
            if f['extract_supergrams'] == 'True':
                terms_so_far.append(supergrams)
            if f['extract_supergrams'] == 'False':
                pass
            terms_so_far = [t for t_sublist in terms_so_far for t in t_sublist]
            for x in terms_so_far:
                all_terms.append(x)
        all_terms = list(set(all_terms))
        terms_n_tfidf = calculate_tf_idf(doc_for_tf_idf, MAX_LEN_NGRAM, list(set(all_terms)))
        all_abvs = [abv for abvs_sublist in all_abvs for abv in abvs_sublist]
        termTime = time.time()
        logging.basicConfig()
        logging.info('Terms extracted in ' + str(termTime - start))
        whitelisted_terms = crosscheck_white_black_lists(all_terms, WHITELIST, BLACKLIST)
        for term in whitelisted_terms:
            if term in terms_n_tfidf.keys():
                continue
            else:
                terms_n_tfidf.update({term: 1.0})
        for abv in all_abvs:
            terms_n_tfidf.update({abv: 1.0})
        return terms_n_tfidf
    
    def add_term_to_cas(self, cas_view, sentence, np):
        import ahocorasick as ahc
        A = ahc.Automaton()
        Token = TYPESYSTEM.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
        A.add_word(np.text, (1, np.text.strip()))
        A.make_automaton()
        for tag in cas_view.select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
            if tag.get_covered_text() == sentence.get_covered_text():
                for end_index, (tfidf, term) in A.iter(sentence.get_covered_text()):
                    start_index = end_index - len(term) + 1
                    cas_view.add_annotation(
                        Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=1.0, term=np.text))
        
    def check_if_term_annotated(self,cas_view, sentence, np):
        terms = [token.get_covered_text() for token in cas_view.select_covered("de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf", sentence)]
        if np.text not in terms:
            return False
        else:
            return True
        
    def check_definitions(self, cas):
        import string
        import re
        cas_view = cas.get_view("html2textView")
        for sentence in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
            definition = sentence.get_covered_text()
            definition = re.sub(r'\(\w{1}\)', '', definition)
            definition = definition.translate(str.maketrans('', '', string.punctuation))
            definition = definition.strip()

            doc = NLP(definition)
            for np in doc.noun_chunks:
                printed_np = False
                for token in np:
                    if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                        term_has_been_annotated = self.check_if_term_annotated(cas_view, sentence, np)
                        if term_has_been_annotated:
                            continue
                        else:
                            self.add_term_to_cas(cas_view, sentence, np)

        return cas

    def post(self, request):
        start = time.time()

        f = request.data

        try:
            decoded_cas_content = base64.b64decode(f['cas_content']).decode('utf-8')
        except binascii.Error:
            logging.info(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")

        cas = load_cas_from_xmi(decoded_cas_content, typesystem=TYPESYSTEM)

        try:
            cas.get_view("html2textView")
        except:
            logging.info(f"could not process the view in this CAS. Make sure it is html2textView.")
            return JsonResponse(f)

        sofa_id = "html2textView"
        sentences = get_text(cas, sofa_id, tagnames=['p'])
        terms_n_tfidf = self.launch_term_extraction(sentences, f)
        cas = add_terms_and_lemmas_to_cas(NLP, cas, TYPESYSTEM, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])
        print(cas.to_xmi())
        cas = self.check_definitions(cas)
        print(cas.to_xmi())
        
        cas_string = base64.b64encode(bytes(cas.to_xmi(), 'utf-8')).decode()
        end = time.time()
        f['cas_content'] = cas_string
        f.pop('extract_supergrams', None)
        logging.info(end - start)

        print(end - start)
        return JsonResponse(f)
