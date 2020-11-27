import pytest
import configparser
import os

from dgconcepts.pipeline.terms_defined.terms_defined_regex import process_definitions_regex

MEDIA_ROOT='tests/test_files'

config = configparser.ConfigParser()
config.read( os.path.join( MEDIA_ROOT, 'TermExtraction.config' ))

def test_process_definitions_regex():
    
    sentences=[
    "test sentence 'term'",
    "test •term • •term• and 'AA' sentence",
    "some sentence",
    "• test test'f 'something' 'and'  •ABC• more ‧s‧  ‧sss‧ ‧‧df"
    ]
    
    true_indices=[
    [('term', 15, 19)], 
    [('aa', 25, 27), ('term', 14, 18)], 
    [],
    [('abc', 34, 37), ('something', 15, 24), ('and', 27, 30), ('sss', 50, 53)]
    ]
        
    assert true_indices == list(process_definitions_regex( sentences, config ))
    

def test_process_definitions_regex_empty_sentences():
    
    sentences=[
    "" 
    ]
    
    true_indices=[
    []
    ]
        
    assert true_indices == list(process_definitions_regex( sentences, config ))