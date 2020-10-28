from typing import List, Set

from nltk.corpus import stopwords
import contractions
import re

from .metrics import calculate_tf_idf

POS_TAG_DET = 'DET'
INVALID_POS_LABELS = ['ADP', 'VERB', 'PRON', 'CCONJ', 'SCONJ', 'ADV']

# ---------------------
# ABBREVIATION EXTRACTION
# ---------------------
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
    tokens = [t.text for t in tokens]
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
        #abv = x[0] + " ■ " + x[1]
        abvs.append( (x[0].strip(), x[1].strip() ) )

    return abvs


# ---------------------
# TERM EXTRACTION
# ---------------------

def get_ngrams_supergrams(tree, max_ngram_length):
    """

    :param tree: a set of noun chunks derived from the SpaCy Doc object
    :param max_ngram_length: the max ngram length
    :return: 2 lists : ngrams (noun phrases with length <= max_ngram_length) and supergrams (noun phrases with length >= max_ngram_length)
    """
    ngrams = []
    supergrams = []
    for ngram in tree:
        if ngram[0].pos_ == POS_TAG_DET:
            ngram = ngram[1:]
        if len(ngram) > max_ngram_length:
            supergrams.append(ngram.text)
        else:
            if validate_term(ngram):  # grammar check
                ngrams.append(ngram.text)
    
    #TO DO check why ngram not unique
    ngrams=list(set( ngrams ))
    supergrams=list( set(supergrams ) )
                
    return ngrams, supergrams

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


def validate_term(np):
    """
    :param np: noun phrase to be checked for validation
    :return: whether or not the ngram is a valid term
    """

    invalid_words = get_invalid_words()
    if any(word.pos_ in INVALID_POS_LABELS for word in np) == False and all(word.text.isalpha() for word in np) and any(word.text in invalid_words for word in np)==False and len(np.text) > 1:
        return True
    else:
        return False


def clean_text(data):
    """
    :param data: the text segment
    :return: the cleaned text segment
    """
    # clean
    data = data.lower()
    data = data.replace(u'\xa0', ' ')
    data = contractions.fix(data)
    return data


def parse_doc(doc):
    """

    :param doc: SpaCy object Doc
    :return: a set containing noun phrases of type spacy.tokens.span.Span
    """
    tree = {np for nc in doc.noun_chunks for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]}
    return tree


def extract_concepts(text, NLP, nMax):
    """
    :param text: the text segment
    :param NLP: the spacy model
    :param nMax: the max ngram length
    :return: 3 lists : ngrams (n <= nMax), supergrams (n >= nMax) and abbreviations
    """
    clean_data = clean_text(text)
    doc = NLP(clean_data)
    tree = parse_doc(doc)
    abvs = extractAbbv(NLP(text))
    ngrams, supergrams = get_ngrams_supergrams(tree, nMax)
    return ngrams, supergrams, abvs

#Main functions

def get_terms( NLP , sentences: List[str] , extract_supergrams:bool=False, nMax:int=4 ) -> dict:

    terms=[]
    all_abvs = []
    doc_for_tf_idf = []
    for sentence in sentences:
        doc_for_tf_idf.append(sentence)
        ngrams, supergrams, abvs = extract_concepts(sentence, NLP, nMax )
        all_abvs+=abvs
        terms+=ngrams
        if extract_supergrams:
            terms+=supergrams
              
    for abv in all_abvs:
        abbreviation=abv[0].strip()
        full_abbreviation=abv[1].strip()
        if not abbreviation or not full_abbreviation:
            continue
        terms.append( abbreviation.lower() )
        terms.append( full_abbreviation.lower() )

    terms = list(set(terms))
    all_abvs=list( set(all_abvs ))
    terms_n_tfidf = calculate_tf_idf(doc_for_tf_idf, nMax, terms )
    
    return terms_n_tfidf, all_abvs


def get_terms_defined_in_regex(  definitions: List[str] ) -> List[str]:
    
    '''
    to do
    '''
    
    #regex for terms in between quotes
    regex_terms=r"[\‘|\"|\`|\'|\’|\•|\“\‧][a-z0-9\-(){}_/\\]{2,}[a-z0-9 \-(){}_/\\]*[a-z0-9\-(){}_/\\]+[\‘|\"|\`|\'|\’|\•|\”\‧]"

    #regex for abbreviations in between quotes
    regex_abbv=r"[\‘|\"|\`|\'|\’|\•|\“\‧][A-Z]{2,}[\‘|\"|\`|\'|\’|\•|\”\‧]"

    terms=[]
    
    for definition in definitions:

        match_term=re.finditer( regex_terms, definition, re.IGNORECASE  )
        match_abbv=re.finditer( regex_abbv, definition  )

        #first annotate the abbreviations:
        for m in match_abbv:
            if m:
                term=m.group(0)[1:-1].strip()
                if term:
                    terms.append( term.lower() )

        #annotate the terms:
        for m in match_term:
            if m:
                term=m.group(0)[1:-1]
                if term:
                    terms.append( term.lower() )
              
    terms=list(set(terms))
            
    return terms

def remove_add_update_terms_blacklist_whitelist( terms_n_tfidf: dict, whitelist: Set[str], blacklist: Set[str], tf_idf_whitelist: float =-1.0  ) -> dict:
    
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
    