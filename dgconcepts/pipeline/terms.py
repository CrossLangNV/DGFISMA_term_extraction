from nltk.corpus import stopwords
import contractions

#---------------------
#ABBREVIATION EXTRACTION
#---------------------
def extractAbbvTerm(tokens,i, sw):
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
        t = t.text
        prop = sum(1 for c in t if c.isupper())/len(t)
        if(prop > 0.5 
                and len(t) < 6 
                and len(t) > 1 
                and t.lower() not in sw 
                and sum(1 for c in t if c == 'V' or c=='I') != len(t) 
                and t.isalpha()):
            term = extractAbbvTerm(tokens,i, sw)
            if(term is not None):
                res.append((t,term))
    abvs = []
    for x in list(set(res)):
        abv = x[0] + " ■ " + x[1]
        abvs.append(abv)

    return abvs

#---------------------
#TERM EXTRACTION
#---------------------
def parse_tree(tree, nMax):
    """

    :param tree: the spacy parser tree
    :param nMax: the max ngram length
    :return: 2 lists : ngrams (noun phrases with length <= nMax) and supergrams (noun phrases with length >= nMax)
    """
    ngrams = []
    supergrams = []
    for ngram in tree:
        if ngram[0].pos_ == 'DET':
            ngram = ngram[1:]
        if validateTerm(ngram):
            if len(ngram) > nMax:
                supergrams.append(ngram.text)
            else:
             ngrams.append(ngram.text)
        else:
            continue
    return ngrams, supergrams

def validateTerm(np):
    """
    :param np: noun phrase to be checked for validation (ngram)
    :return: whether or not the ngram is a valid term
    """
    labels = ['ADP', 'VERB', 'PRON', 'CCONJ', 'SCONJ']
    if any(word.pos_ in labels for word in np) == False and all(word.text.isalpha() for word in np) and len(np.text) > 1:
        return True
    else:
        return False

def cleanText(data):
    """
    :param data: the text segment
    :return: the cleaned text segment
    """
    # clean
    data = data.lower()
    data = data.replace("a.", '')
    data = data.replace("b.", '')
    data = data.replace("c.", '')
    data = data.replace("d.", '')
    data = data.replace("e.", '')
    data = data.replace("f.", '')
    data = data.replace("g.", '')
    data = data.replace("h.", '')
    data = data.replace("i.", '')
    data = data.replace("j.", '')
    data = data.replace("k.", '')
    data = data.replace("l.", '')
    data = data.replace("m.", '')
    data = data.replace("n.", '')
    data = data.replace("o.", '')
    data = data.replace("p.", '')
    data = data.replace("q.", '')
    data = data.replace("r.", '')
    data = data.replace("s.", '')
    data = data.replace("t.", '')
    data = data.replace("u.", '')
    data = data.replace("v.", '')
    data = data.replace("w.", '')
    data = data.replace("x.", '')
    data = data.replace("y.", '')
    data = data.replace("z.", '')
    data = data.replace("ii", '')
    data = data.replace("iii", '')
    data = data.replace("iv", '')
    data = data.replace("vi", '')
    data = data.replace("vii", '')
    data = data.replace("viii", '')
    data = data.replace("oj", '')
    data = data.replace("ix", '')
    data = data.replace("xi", '')
    data = contractions.fix(data)
    return data

#---------------------
#MAIN FUNCTION
#---------------------
def extractConcepts(text, NLP, nMax):
    """
    :param text: the text segment
    :param NLP: the spacy model
    :param nMax: the max ngram length
    :return: 3 lists : ngrams (n <= nMax), supergrams (n >= nMax) and abbreviations
    """
    clean_data = cleanText(text)
    doc = NLP(clean_data)
    abvs = extractAbbv(doc)
    tree = {np for nc in doc.noun_chunks for np in [nc,doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]}
    ngrams, supergrams = parse_tree(tree, nMax)
    return ngrams, supergrams, abvs
