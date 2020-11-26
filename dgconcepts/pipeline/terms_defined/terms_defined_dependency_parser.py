from typing import Generator, List, Tuple

from cassis import Cas, TypeSystem

from spacy.lang.en import English      
        
def process_definitions_dependency_parser(  sentences: List[str], nlp: English ) -> Generator[ List[ Tuple[ str, int, int ] ], None, None ] :
        
    '''
    Dependency parsing of sentences.
    Function returns a Generator yielding a list of terms (dependend/defined_terms) and offsets for each sentence.
    
    :param sentences: List. List of Strings. 
    :param NLP: spacy model. Spacy model used for dependency parsing.
    :return: Generator.  
    '''
        
    for sentence in sentences:
        detected_terms=[]
        doc=nlp(  sentence  )
        for sent in doc.sents:
            for np in sent.noun_chunks:
                for token in np:
                    if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                        detected_terms.append( ( token.text  , token.idx , token.idx + len(token)  ) )
                        break
        yield detected_terms
        