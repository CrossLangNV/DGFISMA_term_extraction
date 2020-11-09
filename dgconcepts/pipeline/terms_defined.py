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
            #case where tf_idf term is found via regex and has a good dependency type
            if (tf_idf.tfidfValue == tf_idf_regex and defined_term(tf_idf)=='nsubj' ): #make this configurable
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True
                    
        #if one of the terms found via regex is considered the term confirmed via dependency parsing, stop searching.
        if defined_detected:
            continue

        for tf_idf in terms_sentence:
            #case where tf_idf term is found via regex and does not have a bad dependency type
            if (tf_idf.tfidfValue == tf_idf_regex and defined_term(tf_idf)!='bad' ): 
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True   
                    
        #if one of the terms found via regex is not considered bad, stop searching
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

'''
#new approach, failing on some cases

from cassis import Cas, TypeSystem

from spacy.lang.en import English      
                    
def add_nsubj_dependency( nlp: English, cas:Cas, typesystem: TypeSystem , SofaID:str, \
                         definition_type:str='de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence', \
                         dependency_type:str="de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency" ):
    
    dpdc = typesystem.get_type( dependency_type )
    
    for sentence in cas.get_view( SofaID ).select( definition_type ):

        doc=nlp(  sentence.get_covered_text()  )
        for sent in doc.sents:
            passive_voice=False

            for np in sent.noun_chunks:
                for token in np:
                    if token.dep_ == 'nsubjpass':
                        passive_voice=True

                        cas.get_view( SofaID ).add_annotation( dpdc( begin=sentence.begin+token.idx, end=(sentence.begin+token.idx+len(token)), \
                                                                    DependencyType='defined_passive' ))

            if passive_voice:
                for np in sent.noun_chunks:
                    for token in np:
                        if token.dep_ == 'nsubj':
                            cas.get_view( SofaID ).add_annotation( dpdc( begin=sentence.begin+token.idx, end=(sentence.begin+token.idx+len(token)), \
                                                                    DependencyType='definition_passive' ))

            #the sentence is active
            else:
                for token in sent.root.children:
                    if token.dep_=='nsubj':
                        cas.get_view( SofaID ).add_annotation( dpdc( begin=sentence.begin+token.idx, end=(sentence.begin+token.idx+len(token)), \
                                                                    DependencyType='defined_active' ))

                    elif token.dep_=='dobj':
                        cas.get_view( SofaID ).add_annotation( dpdc( begin=sentence.begin+token.idx, end=(sentence.begin+token.idx+len(token)), \
                                                                    DependencyType='definition_active' ))
                        
                                     
def add_defined_term( cas: Cas, typesystem: TypeSystem, SofaID:str, definition_type:str = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' ,\
                      token_type:str = 'de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf', \
                      defined_type:str = 'cassis.Token', \
                      dependency_type:str = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", \
                      tf_idf_whitelist:float = -1.0, \
                      tf_idf_regex:float = -2.0, \
                      tf_idf_threshold:float = -9999.0):
    
    def defined_term( tf_idf ):
        
        
        for dependency in cas.get_view( SofaID ).select_covered( dependency_type, tf_idf ):
            dependency_tag=dependency.DependencyType
            
            if dependency_tag == 'definition_passive' or dependency_tag == 'definition_active':
                return 'definition'
            elif dependency_tag == 'defined_active' or dependency_tag == 'defined_passive':
                return 'defined'
        
        return 'no_definition_no_defined'
    
    Token = typesystem.get_type( defined_type )
    
    #iteration over the definitions
    for definition in cas.get_view( SofaID ).select( definition_type ):
    
        defined_detected=False
        
        terms_sentence = list(cas.get_view( SofaID ).select_covered( token_type, definition  ) )
        
        for tf_idf in terms_sentence:
            #case where tf_idf term is found via regex and has a good dependency type
            if (tf_idf.tfidfValue == tf_idf_regex and defined_term(tf_idf)=='defined' ): #make this configurable
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True
                    
        #if one of the terms found via regex is considered the term confirmed via dependency parsing, stop searching.
        if defined_detected:
            continue

        for tf_idf in terms_sentence:
            #case where tf_idf term is found via regex and does not have a bad dependency type
            if (tf_idf.tfidfValue == tf_idf_regex and defined_term(tf_idf)!='definition' ): 
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True   
                    
        #if one of the terms found via regex is not considered bad, stop searching
        if defined_detected:
            continue
          
        for tf_idf in terms_sentence:
            #case where tf_idf term is whitelisted and has a good dependency type
            if (tf_idf.tfidfValue == tf_idf_whitelist and defined_term(tf_idf)=='defined' ): #make this configurable
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                    defined_detected=True
            
        #if one of the whitelisted terms is considered the defined term, stop searching
        if defined_detected:
            continue
            
        #case when there are not whitelisted/regex terms in the sentence, and/or all whitelisted terms were rejected by dependency parser 
        for tf_idf in terms_sentence:
            if defined_term( tf_idf ) =='defined' and tf_idf.tfidfValue > tf_idf_threshold:
                cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin  , end=tf_idf.end ) )
                defined_detected=True
                
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
'''