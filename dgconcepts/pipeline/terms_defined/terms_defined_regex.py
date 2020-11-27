from typing import Generator, List, Tuple
from configparser import ConfigParser

import re

def process_definitions_regex(  sentences: List[str], config: ConfigParser ) -> Generator[ List[ Tuple[ str, int, int ] ], None, None ]:
    
    '''
    Function finds position of terms via a regex. Regex finds terms in between quotes.
    Function returns a Generator yielding a list of terms (dependend/defined_terms) and offsets for each sentence.
    
    :param sentences: List. List of Strings. 
    :param config: ConfigParser.
    :return: Generator.  
    '''
    
    #regex for terms in between quotes
    regex_terms=config[ "Regex" ].get( "REGEX_TERMS" )
    #r"[\‘|\"|\`|\'|\’|\•|\“|\‧][a-z0-9\-(){}_/\\]{2,}[a-z0-9 \-(){}_/\\]*[a-z0-9\-(){}_/\\]+[\‘|\"|\`|\'|\’|\•|\”|\‧]"

    #regex for abbreviations in between quotes
    regex_abbv=config[ "Regex" ].get( "REGEX_ABBREVIATIONS" )  
    #r"[\‘|\"|\`|\'|\’|\•|\“|\‧][A-Z]{2,}[\‘|\"|\`|\'|\’|\•|\”|\‧]"

    terms=[]
    
    for sentence in sentences:

        detected_terms=[]
        
        match_term=re.finditer( regex_terms, sentence, re.IGNORECASE  )
        match_abbv=re.finditer( regex_abbv, sentence )

        #get the abbreviations:
        for m in match_abbv:
            if m and len( m.group(0))>2:
                term=m.group(0)[1:-1]
                if term:
                    detected_terms.append( ( term.lower(), m.span()[0]+1, m.span()[1]-1 ) )
                
        #get the terms:
        for m in match_term:
            if m and len( m.group(0))>2:
                term=m.group(0)[1:-1]
                if term:
                    detected_term=( term.lower(), m.span()[0]+1, m.span()[1]-1 )
                    if detected_term not in detected_terms:  #check if not already detected via abbreviations regex
                        detected_terms.append( detected_term )
              
        yield detected_terms
