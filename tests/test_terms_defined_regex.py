import pytest

from dgconcepts.pipeline.terms_defined_regex import process_definitions_regex

def test_process_definitions_regex():
    
    sentences=[
    "test sentence 'term'",
    "test •term • •term• and 'AA' sentence",
    "some sentence" 
    ]
    
    true_indices=[
    [('term', 15, 19)], 
    [('aa', 25, 27), ('term', 14, 18)], 
    []
    ]
        
    assert true_indices == list(process_definitions_regex( sentences ))
    

def test_process_definitions_regex_empty_sentences():
    
    sentences=[
    "" 
    ]
    
    true_indices=[
    []
    ]
        
    assert true_indices == list(process_definitions_regex( sentences ))