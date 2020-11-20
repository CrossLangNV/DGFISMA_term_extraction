import os

from cassis import Cas
from cassis.typesystem import load_typesystem

import configparser

MEDIA_ROOT='media'
                        
with open( os.path.join( MEDIA_ROOT, 'typesystem.xml' )  , 'rb') as f:
    TYPESYSTEM = load_typesystem(f)

config = configparser.ConfigParser()
config.read( os.path.join( MEDIA_ROOT, 'TermExtraction.config' ))

def _create_test_cas_1( extra_dependencies=True ):
    
    cas = Cas( typesystem=TYPESYSTEM )

    definitions=[
    "'this' or 'that this' means something this",
    "this or that and 'more'",
    "some sentence",
    ""
    ]

    cas.create_view( config[ 'Annotation' ]['Sofa_ID'] )
    cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).sofa_string = "\n".join(definitions)

    Sentence_type=TYPESYSTEM.get_type( config[ 'Annotation' ][ 'DEFINITION_TYPE' ] )

    Token_type=TYPESYSTEM.get_type(  config[ 'Annotation' ][ 'TOKEN_TYPE' ] )
    Dependency_type=TYPESYSTEM.get_type(  config[ 'Annotation' ][ 'DEPENDENCY_TYPE' ]  )

    #Only annotate sentences that are definitions
    pos=0
    for definition in definitions:
        cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).add_annotation( Sentence_type(begin=pos , end=pos+len( definition ) , id="definition"  ) )  
        pos=pos+len( definition )+1

    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="this", begin=1 , end=5, tfidfValue=-2   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="this", begin=16 , end=20, tfidfValue=-2   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="this", begin=38 , end=42, tfidfValue=-2   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="this", begin=43 , end=47, tfidfValue=-2   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="that this", begin=11 , end=20, tfidfValue=-2   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="more", begin=61 , end=65, tfidfValue=-2  ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="that", begin=51 , end=55, tfidfValue=-1   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="that and", begin=51 , end=59, tfidfValue=-1   ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="something", begin=28 , end=37, tfidfValue=0.23  ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="some", begin=67 , end=71, tfidfValue=0.74  ) )
    cas.get_view(   config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Token_type( term="some sentence", begin=67 , end=80, tfidfValue=0.72  ) )

    #some sanity check if added correctly
    for term in cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'TOKEN_TYPE' ] ):
        assert term.term == term.get_covered_text()

    #add dummy dependency annotation
    cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Dependency_type( begin=16 , end=20 , DependencyType='nsubj' ) )        
    if extra_dependencies:
        cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Dependency_type( begin=52 , end=54 , DependencyType='nsubj' ) )
        cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).add_annotation(  Dependency_type( begin=67 , end=71 , DependencyType='nsubj' ) )
    
    return cas
