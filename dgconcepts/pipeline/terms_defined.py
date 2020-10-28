from cassis import Cas, TypeSystem

from spacy.lang.en import English      
    
    
def add_nsubj_dependency( nlp: English, cas:Cas, typesystem: TypeSystem , SofaID:str, \
                         definition_type:str='de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence', \
                         dependency_type:str="de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency" ):
    
    dpdc = typesystem.get_type( dependency_type )
    
    for sentence in cas.get_view( SofaID ).select( definition_type ):

        doc=nlp( sentence.get_covered_text() )
        for np in doc.noun_chunks:
            for token in np:
                if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                    cas.get_view( SofaID ).add_annotation( dpdc( begin=sentence.begin+token.idx, end=(sentence.begin+token.idx+len(token)), \
                                                               DependencyType='nsubj' ))
                    
                    
def add_defined_term( cas: Cas, typesystem: TypeSystem, SofaID:str, definition_type:str = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' ,\
                      token_type:str = 'de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf', \
                      defined_type:str = 'cassis.Token', \
                      dependency_type:str = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", \
                      tf_idf_whitelist:float = -1.0, \
                      tf_idf_regex:float = -2.0, \
                      tf_idf_threshold:float = -9999.0):
    
    def defined_term( tf_idf ):
        
        '''helper function to find dependency relation of the tfidf annotation'''
        
        for dependency in cas.get_view( SofaID ).select_covered( dependency_type, tf_idf ):
            if dependency.DependencyType == 'bad':
                return 'bad'
            elif dependency.DependencyType == 'nsubj':
                return 'nsubj'
        
        return ''
    
    Token = typesystem.get_type( defined_type )
    
    #iteration over the definitions
    for definition in cas.get_view( SofaID ).select( definition_type ):
    
        defined_detected=False
        
        terms_sentence = list(cas.get_view( SofaID ).select_covered( token_type, definition  ) )
        
        for tf_idf in terms_sentence:
            #case where tf_idf term is found via regex and does not have a bad dependency type
            #if (tf_idf.tfidfValue == whitelist_score_tf_idf and defined_term(tf_idf)=='nsubj' ): #make this configurable   
            if (tf_idf.tfidfValue == tf_idf_regex and defined_term(tf_idf)!='bad' ): #make this configurable
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True

        #if one of the terms found via regex is considered the defined term, stop searching
        if defined_detected:
            continue
          
        for tf_idf in terms_sentence:
            #case where tf_idf term is whitelisted and has a good dependency type
            if (tf_idf.tfidfValue == tf_idf_whitelist and defined_term(tf_idf)=='nsubj' ): #make this configurable
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True
            
        #if one of the whitelisted terms is considered the defined term, stop searching
        if defined_detected:
            continue
            
        #case when there are not whitelisted/regex terms in the sentence, and/or all whitelisted terms were rejected by dependency parser 
        for tf_idf in terms_sentence:
            if defined_term( tf_idf ) =='nsubj' and tf_idf.tfidfValue > tf_idf_threshold:
                cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin  , end=tf_idf.end ) )
                defined_detected=True