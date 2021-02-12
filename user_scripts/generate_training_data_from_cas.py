from typing import List, Tuple, Set
import glob
import os

from cassis import Cas, TypeSystem
from cassis.xmi import load_cas_from_xmi
from cassis.typesystem import load_typesystem

import configparser
from configparser import ConfigParser

from dgconcepts.pipeline.utils import deepest_child
from .utils import remove_quotations_around_terms

from pathlib import Path
import plac

def extract_user_annotations_from_cas( cas: Cas, config: ConfigParser, users:Set[str]={} ) -> Tuple[List[str],List[str]]:

    '''
    Function to extract annotations added by the user in the CAS. Extracts terms, definitions + defined token.
    
    :param cas: Cas. 
    :param config: ConfigParser.
    :param user: Set. List of users of which annotations should be extracted.
    :return: Tuple[List[str],List[str]]. A list of terms, definitions, and a list of annotated definitions (i.e. the tokens that are defined by the definition).
    '''
    
    def sanity_check( defined ):
        #Helper function to check that annotations added by considered users do not overlap
        previous_index=0
        for item in defined:
            if item[0] < previous_index:
                return False
            previous_index=item[1]
        return True
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    value_between_tagtype=config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )
    token_type_user=config['Annotation_user']['TOKEN_TYPE_USER']
    defined_type_user=config['Annotation_user']['DEFINED_TYPE_USER']

    terms=[]
    annotated_sentences=[]
    definitions=[]
    users=set( users )
    tagnames=set('p')
    
    for token in cas.get_view(SofaID ).select( token_type_user ):
        if (token.user in users or not users):
            terms.append( token.term )
            
    for tag in cas.get_view( SofaID  ).select( value_between_tagtype ):
        if tag.tagName not in tagnames:
            continue
        if deepest_child(  cas, SofaID , tag, tagnames=tagnames , value_between_tagtype=value_between_tagtype ):
            tokens=list( cas.get_view( SofaID ).select_covered(  defined_type_user , tag ) )
            if tokens:
                sentence=tag.get_covered_text()
                defined=[ ( token.begin - tag.begin , token.end - tag.begin ) for token in tokens if (token.user in users or not users ) ]
                if defined:
                    definitions.append(sentence)                
                if not sanity_check( defined ):
                    print( f"The sentence {tag.get_covered_text()} contains overlapping user annotations '{tokens}' added by users: {users}. \
                          Please make sure annotations added by users '{users}' do not overlap."  )
                    continue
                annotated_sentences.append( annotations_to_tags( sentence, defined ) )

    annotated_sentences=[ remove_quotations_around_terms( sentence ) for sentence in  annotated_sentences ]

    return terms, definitions, annotated_sentences


def annotations_to_tags( sentence:str, defined:List[ Tuple[ int , int ] ] , tag_begin="★", tag_end="☆"  )-> str:

    '''
    Given a sentence, and a list of Tuples representing offsets of token annnotations for defined terms, this function converts the sentence to a sentence with annotations that can be used as input to the generate_training_data.py user script.
    '''
    
    sentence_new=''
        
    for i in range( len(defined) ):
        if i==0:
            sentence_new=sentence[ :defined[i][0] ]
        sentence_new=sentence_new + f" {tag_begin} " + sentence[ defined[i][0]:defined[i][1] ] + f" {tag_end} "
        if i<len(defined)-1:
            sentence_new=sentence_new + sentence[ defined[i][1] : defined[i+1][0] ]
        else:
            sentence_new=sentence_new + sentence[ defined[i][1] :]
        
    return sentence_new


@plac.annotations(
    #input-output
    path_annotated_cas_files=( "Path to folder with CAS objects containing user annotations. (I.e. folder containing .xmi files).", ),
    path_typesystem=( "Path to Typesystem that can be used for loading the cas.", ),
    path_config=( "Path to Configuration file.", ),
    path_output_file_terms=( "Path to the output file with terms (.txt).", ),
    path_output_file_definitions=( "Path to the output file with definitions (.txt).", ),
    path_output_file_defined_tokens=( "Path to the output file with annotated tokens (terms that are defined) (.txt),serves as input for user script 'generate_training_data' (.txt)", ),
    users=( "Users from which annotations are considered." , "option" ),
    tag_begin=( "Tag (utf-8 character) used for annotating begin of defined term. Make sure this character is in the bert vocabulary.", "option" ),
    tag_end=( "Tag (utf-8 character) used for annotating end of defined term. Make sure this character is in the bert vocabulary.", "option" ),
)
def main( path_annotated_cas_files:Path,\
          path_typesystem: Path, \
          path_config: Path , \
          path_output_file_terms:Path,\
          path_output_file_definitions:Path,\
          path_output_file_defined_tokens:Path,\
          users:Set[str]={},\
          tag_begin:str="★", \
          tag_end:str="☆" ):
    '''
    Extract training data from CAS
    '''
    
    dirname=os.path.dirname( path_output_file_terms )
    if dirname:
        os.makedirs( dirname , exist_ok=True)
    
    dirname=os.path.dirname( path_output_file_definitions )
    if dirname:
        os.makedirs( dirname , exist_ok=True)
        
    dirname=os.path.dirname( path_output_file_defined_tokens )
    if dirname:
        os.makedirs( dirname , exist_ok=True)
    
    with open( path_typesystem , 'rb') as f:
        TYPESYSTEM = load_typesystem(f)
        
    CONFIG = configparser.ConfigParser()
    CONFIG.read( path_config )
    
    xmi_files=glob.glob( os.path.join( path_annotated_cas_files , "*.xmi"  ) )
    with open( path_output_file_terms, "w" ) as w:
        with open( path_output_file_definitions , "w"  ) as f:
            with open( path_output_file_defined_tokens, 'w'  ) as g:
                for xmi_file in xmi_files:
                    cas_file=open( xmi_file , "rb" )
                    cas=load_cas_from_xmi( cas_file, typesystem=TYPESYSTEM, trusted=True )
                    terms, definitions, annotated_sentences =extract_user_annotations_from_cas( cas, CONFIG, users )
                    for term in terms:
                        w.write( f"{term}\n" )
                    for definition in definitions:
                        f.write( f"{definition}\n"  )
                    for annotated_sentence in annotated_sentences:
                        g.write( f"{annotated_sentence}\n" )