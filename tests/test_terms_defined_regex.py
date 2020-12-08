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
    "test • term• •term• and 'AA' sentence",
    "some sentence",
    "• test test'f 'something' 'and'  •ABC• more ‧s‧  ‧sss‧ ‧‧df",
    '‘management company’s host member state’: some   "term"',
    '‘management company’s host member state’ some "term"  ',
    '‘management company’s host member state’": some   "term" ',
    '‘management company’s host member state""": some   "term" ',
    '"the equity method in the investor’s share": test ',
    '"the equity method in the investor’s share" test ',
    "'AA’s' ",
    "test 'AAs'",    
    ]
    
    true_indices=[
    [('term', 15, 19)], 
    [('aa', 25, 27), ('term', 14, 18)], 
    [('aa', 25, 27), ('term', 14, 18)], 
    [],
    [('abc', 34, 37), ('something', 15, 24), ('and', 27, 30), ('sss', 50, 53)],
    [ (  'management company’s host member state' , 1,39 ) , ('term', 50, 54 ) ],
    [ (  'management company’s host member state' , 1,39 ) , ('term', 47, 51 ) ],
    [('management company’s host member state’', 1, 40), ('term', 51, 55)],
    [('management company’s host member state', 1, 39), ('term', 52, 56)],
    [('the equity method in the investor’s share', 1, 42)],
    [('the equity method in the investor’s share', 1, 42)],
    [('aa’s', 1, 5)],
    [('aas', 6, 9)]    
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