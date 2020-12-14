from typing import List, Set, Dict, Tuple

from spacy.lang.en import English

from nltk.corpus import stopwords
import contractions
import re

from .metrics import calculate_tf_idf

class TermExtractor():
    '''
    A TermExtractor.
    '''
    
    POS_TAG_DET = 'DET'
    INVALID_POS_LABELS = ['ADP', 'VERB', 'PRON', 'CCONJ', 'SCONJ', 'ADV']
    
    def __init__( self, spacy_model: English, extract_supergrams:bool=False, nMax:int=4 ):
        
        '''
        :param spacy_model: English. Spacy model.
        :param extract_supergrams: Bool.
        :param nMax: int. Max length of extracted n-grams.
        '''

        self._nlp=spacy_model
        self._extract_supergrams=extract_supergrams
        self._nMax=nMax
                
    def get_terms( self, sentences: List[str] , n_jobs=12, batch_size=32 )-> Tuple[Dict, List[Tuple[str,str]]]:

        '''
        Function uses spacy model to extract nouns from a list of sentences.

        :param sentences: List. List of strings 
        :param n_jobs: Int. Nr of jobs spacy model.
        :param batch_size: batch_size used by spacy model. 
        :return: Tuple. Dictionary with detected terms and tf_idf score, and a list of abbreviations.
        '''

        terms=[]
        all_abvs = []
        
        #extract terms
        clean_sentences= [clean_text( sentence ) for sentence in sentences ]
                    
        trees=[]
        for doc in self._nlp.pipe( clean_sentences, n_process=n_jobs, batch_size=batch_size ):
            trees.append(self.parse_doc(doc))
        
        for tree in trees:
            ngrams, supergrams = self.get_ngrams_supergrams(tree)
            terms+=ngrams
            if self._extract_supergrams:
                terms+=supergrams
            
        for doc_abv in self._nlp.pipe( sentences, n_process=n_jobs, batch_size=batch_size ):        
            all_abvs+=self.extractAbbv(doc_abv)
      
        #add abbreviations to extracted terms
        for abv in all_abvs:
            abbreviation=abv[0].strip()
            full_abbreviation=abv[1].strip()
            if not abbreviation or not full_abbreviation:
                continue
            terms.append( abbreviation.lower() )
            terms.append( full_abbreviation.lower() )

        terms = list(set(terms))
        all_abvs=list( set(all_abvs ))
        terms_n_tfidf = calculate_tf_idf(sentences, self._nMax, terms )

        return terms_n_tfidf, all_abvs
        
    def get_ngrams_supergrams(self, tree)->Tuple[ List[str],List[str] ]:
        
        '''
        
        :param tree: a set of noun chunks derived from the SpaCy Doc object
        :return: 2 lists : ngrams (noun phrases with length <= max_ngram_length) and supergrams (noun phrases with length >= max_ngram_length)
        '''
        
        ngrams = []
        supergrams = []
        for ngram in tree:
            if ngram[0].pos_ == self.POS_TAG_DET:
                ngram = ngram[1:]
            if len(ngram) >  self._nMax:  #add check self._extract_supergrams
                supergrams.append(ngram.text)
            else:
                if self.validate_term(ngram):  # grammar check
                    ngrams.append(ngram.text)

        #TO DO check why ngram not unique
        ngrams=list(set( ngrams ))
        supergrams=list( set(supergrams ) )

        return ngrams, supergrams

    def validate_term(self, np)->bool:
        
        '''
        
        :param np: noun phrase to be checked for validation
        :return: whether or not the ngram is a valid term
        '''
        invalid_words = get_invalid_words()
        if any(word.pos_ in self.INVALID_POS_LABELS for word in np) == False and all(word.text.isalpha() for word in np) and any(word.text in invalid_words for word in np)==False and len(np.text) > 1:
            return True
        else:
            return False
        
    def parse_doc(self, doc) -> Set:
        '''
        
        :param doc: SpaCy object Doc
        :return: a set containing noun phrases of type spacy.tokens.span.Span
        '''
        
        tree = {np for nc in doc.noun_chunks for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]}
        return tree
    
    def extractAbbv(self, doc)->List[Tuple[ str, str ]]:
        '''
        
        :param doc: SpaCy object Doc
        :return: A List of Tuples containing abbreviations.
        '''

        tokens = [t.text for t in doc]
        sw = set(stopwords.words('english'))
        res = []
        for i, t in enumerate(tokens):
            prop = sum(1 for c in t if c.isupper()) / len(t)
            if (prop > 0.5
                    and len(t) < 6
                    and len(t) > 1
                    and t.lower() not in sw
                    and sum(1 for c in t if c == 'V' or c == 'I') != len(t)
                    and t.isalpha()):
                term = extractAbbvTerm(tokens, i, sw)
                if (term is not None):
                    res.append((t, term))
        abvs = []
        for x in list(set(res)):
            abvs.append( (x[0].strip(), x[1].strip() ) )

        return abvs


def remove_add_update_terms_blacklist_whitelist( terms_n_tfidf: dict, whitelist: Set[str], blacklist: Set[str], tf_idf_whitelist: float =-1.0  ) -> Dict:
    
    '''
    Functions updates dictionary with detected terms as keys and tf_idf scores as values, using a blacklist and whitelist. Blacklisted terms are removed, whitelisted terms are given a tf_idf_score equal to tf_idf_whitelist.
    
    :param terms_n_tfidf: Dict. 
    :param whitelist: List. List of strings 
    :param blacklist: Bool.
    :param tf_idf_whitelist: float.
    :return: Dict. Update of the terms_n_tfidf dictionary.
    '''
    
    keys=set( terms_n_tfidf.keys() )
    
    #remove blacklisted terms, update tfidf score of whitelisted terms
    for term in keys:
        if term in blacklist:
            terms_n_tfidf.pop( term )
            
    #update tfidf score of whitelisted terms, add whitelisted terms not in blacklist
    difference = list( whitelist - blacklist )
    for term in difference:
        terms_n_tfidf.update( { term: tf_idf_whitelist }  )
            
    return terms_n_tfidf



#helper functions abbreviation extraction:

def extractAbbvTerm(tokens, i, sw):
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
    for j, c in enumerate(abbv[::-1]):
        while (i - j - k >= 0 and tokens[i - j - k][0] != c and c.isupper() and (
                tokens[i - j - k] in sw or tokens[i - j - k] == '’' or tokens[i - j - k][0] == '(')):
            k += 1
        if (i - j - k < 0 or (c.isupper() and tokens[i - j - k][0].lower() != c.lower())):
            return None

    res = ""
    for t in tokens[i - len(abbv) - k + 1:i]:
        if (t != "’" and t != "(" and t != "{" and t != "["):
            res += t + " "
    return res

#helper functions term extraction

def get_invalid_words():
    """
    :return: the list of invalid words for ngram filtering
    """
    invalid_words = ['other', 'such', 'same', 'similar', 'different', 'relevant', 'specific', 'total', 'appropriate'] + stopwords.words('english')
    invalid_words.append("less")
    invalid_words.append("overall")
    invalid_words.append("more")
    invalid_words.append("regardless")
    invalid_words.append("without")
    invalid_words.append("due")
    invalid_words.append("thereof")
    invalid_words.append("please")
    invalid_words.append("with")
    invalid_words.append("also")
    invalid_words.append("would")
    invalid_words.append("whose")
    invalid_words.append("yet")
    invalid_words.append("is")
    invalid_words.append("'s")
    invalid_words.append("(")
    invalid_words.append(")")
    invalid_words.append("[")
    invalid_words.append("]")
    invalid_words.append("{")
    invalid_words.append("}")
    invalid_words.append("be")
    invalid_words.append("to")
    invalid_words.append("fields")
    invalid_words.append("field")
    invalid_words.append("sections")
    invalid_words.append("chapter")
    invalid_words.append("chapters")
    invalid_words.append("section")
    invalid_words.append("articles")
    invalid_words.append("article")
    invalid_words.append("table")
    invalid_words.append("annex")
    invalid_words.append("shall")
    invalid_words.append("whether")
    invalid_words.append("subparagraph")
    invalid_words.append("paragraph")
    invalid_words.append("where")
    invalid_words.append("referred")
    invalid_words.append("within")
    invalid_words.append("may")
    invalid_words.append("is")
    invalid_words.append("not")
    invalid_words.append("new")
    invalid_words.append("old")
    invalid_words.append("according")
    invalid_words.append("accordance")
    invalid_words.append("qualify")
    invalid_words.append("whereas")
    invalid_words.append("therefore")
    invalid_words.append("except")
    invalid_words.append("hereto")
    invalid_words.append("where")
    invalid_words.append("which")
    invalid_words.append("whereof")
    invalid_words.append("since")
    invalid_words.append("hereby")
    invalid_words.append("january")
    invalid_words.append("february")
    invalid_words.append("march")
    invalid_words.append("april")
    invalid_words.append("may")
    invalid_words.append("june")
    invalid_words.append("july")
    invalid_words.append("august")
    invalid_words.append("september")
    invalid_words.append("october")
    invalid_words.append("november")
    invalid_words.append("december")
    invalid_words.append("one")
    invalid_words.append("two")
    invalid_words.append("three")
    invalid_words.append("four")
    invalid_words.append("five")
    invalid_words.append("six")
    invalid_words.append("seven")
    invalid_words.append("eight")
    invalid_words.append("nine")
    invalid_words.append("ten")
    invalid_words.append("''")
    invalid_words.append("``")
    invalid_words.append("`")
    invalid_words.append("(-)")
    invalid_words.append("a.")
    invalid_words.append("b.")
    invalid_words.append("c.")
    invalid_words.append("d.")
    invalid_words.append("e.")
    invalid_words.append("f.")
    invalid_words.append("g.")
    invalid_words.append("h.")
    invalid_words.append("i.")
    invalid_words.append("j.")
    invalid_words.append("k.")
    invalid_words.append("l.")
    invalid_words.append("m.")
    invalid_words.append("n.")
    invalid_words.append("o.")
    invalid_words.append("p.")
    invalid_words.append("q.")
    invalid_words.append("r.")
    invalid_words.append("s.")
    invalid_words.append("t.")
    invalid_words.append("u.")
    invalid_words.append("v.")
    invalid_words.append("w.")
    invalid_words.append("x.")
    invalid_words.append("y.")
    invalid_words.append("z.")
    invalid_words.append("'")
    invalid_words.append("\"")
    invalid_words.append("ii")
    invalid_words.append("iii")
    invalid_words.append("iv")
    invalid_words.append("vi")
    invalid_words.append("vii")
    invalid_words.append("viii")
    invalid_words.append("oj")
    invalid_words.append("ix")
    invalid_words.append("xi")
    invalid_words.append("third")
    invalid_words.append("fourth")
    invalid_words.append("cr")
    invalid_words.append("equ")
    invalid_words.append("irb")
    invalid_words.append("columns")
    invalid_words.append("column")
    invalid_words.append("rows")
    invalid_words.append("row")
    invalid_words.append("item")
    invalid_words.append("items")
    invalid_words.append("point")
    invalid_words.append("points")
    invalid_words.append("part")
    invalid_words.append("parts")
    invalid_words.append("kind")
    invalid_words.append("kinds")
    invalid_words.append("whole")
    invalid_words.append("best")
    invalid_words.append("worst")
    
    return invalid_words

def clean_text(data):
    '''
    :param data: the text segment
    :return: the cleaned text segment
    '''
    # clean
    data = data.lower()
    data = data.replace(u'\xa0', ' ')
    data = contractions.fix(data)
    return data
