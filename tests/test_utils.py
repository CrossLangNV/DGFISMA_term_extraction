import pytest

from dgconcepts.pipeline.utils import is_token

def test_is_token():
    
    texts=[
    "test term text",
    "test termtext",
    "test termtext",
    "test-termtext",
    "term termtext",
    "term-termtext",
    "term-termtext",
    "term-termtext",
    "f g",
    "fg "
    ]
    
    start_end_indices=[
    (5,8),  #"term"
    (5,8),
    (5,12),
    (5,12),
    (0,3),
    (0,3),
    (0,0),
    (0,12),
    (2,2),
    (1,1)
    ]
    
    true_results=[
    True,
    False,
    True,
    False,
    True,
    False,
    False,
    True,
    True,
    False
    ]
    
    for text, start_end_index, true_result in zip( texts, start_end_indices, true_results ):
        assert true_result==is_token( start_end_index[0], start_end_index[1], text, special_characters=[ "-","_","+"] )
        

def test_is_token_special_characters():
    
    texts=[
    "test-termtext",
    "term-termtext" 
    ]
    
    start_end_indices=[
    (5,12),
    (0,3)
    ]
    
    true_results=[
    True,
    True
    ]
    
    for text, start_end_index, true_result in zip( texts, start_end_indices, true_results ):
        assert true_result==is_token( start_end_index[0], start_end_index[1], text, special_characters=[ "_","+"] )
        