import pytest

from dgconcepts.pipeline.definitions_bio_tagging import get_terms_pos_bio_tags, find_indices_tokenized_term_in_text

#test the function "get_terms_pos_bio_tags"

def test_get_terms_pos_bio_tags_sentence1():
    
    detokenized_sentence=[ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ]
    bio_tags= [ 'B' , 'O', 'O', 'B', 'B', 'I', 'O', 'O', 'O', 'B', 'B' ]
    
    items = list( get_terms_pos_bio_tags( detokenized_sentence, bio_tags ))
    
    assert items == [ ('start', 0, 5), ('position start end', 20, 38), ('start end', 48, 57)  ]
    
def test_get_terms_pos_bio_tags_sentence2():
    
    detokenized_sentence=[ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ]
    bio_tags= [ 'O' , 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O' ]
     
    items = list( get_terms_pos_bio_tags( detokenized_sentence, bio_tags ))
    
    assert items == [ ('start', 29, 34) ]
    
    
def test_get_terms_pos_bio_tags_sentence3():
    
    detokenized_sentence=[ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ]
    bio_tags= [ 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O' ]
    
    items = list( get_terms_pos_bio_tags( detokenized_sentence, bio_tags ))
    
    assert items == [  ]
    
#test the function "find_indices_tokenized_term_in_nontokenized_text"

def test_find_indices_tokenized_term_in_nontokenized_text():
    
    sentences=[ 
    'swaption-shops are swaption    --- shops (swaption () shops)  swaption.shops swaption.shops- swaption shops. swaption? shops swaption^shops swaption - shops',
    'swaption.shops are swaption    --- shops (swaption () shops)  swaption.shopsss swaption.shops- swaption shops. swaption? shops swaption^shops swaption . shops',
    'test swaption shops test. swaption? shops test swaption^shops swaption - shops "swaption shops test',
    'test swaption shops 1 test. swaption? shops test swaption^shops swaption - shops "swaption shops 1 test',
    ''
    ]
    
    tokenized_terms=[
        'swaption - shops',
        'swaption . shops',
        'swaption shops test',
        '"swaption shops 1 test',
        ''
    ]
    
    true_indices=[
        [ ( 'swaption-shops' , 0, 14), ( 'swaption - shops', 140, 156) ],
        [ ( 'swaption.shops' , 0, 14), ( 'swaption.shops', 79, 93 ), ( 'swaption . shops', 142, 158) ],
        [ ( 'swaption shops test' , 5, 24), ( 'swaption shops test', 80, 99) ],
        [ ( 'swaption shops 1 test' , 5, 26), ( 'swaption shops 1 test', 82, 103) ],
        [] 
    ]
    
    for sentence, tokenized_term, true_index in zip( sentences, tokenized_terms, true_indices):
        
        items=find_indices_tokenized_term_in_text( tokenized_term , sentence )

        items=[ (item.group(0), item.span()[0], item.span()[1]) for item in items]

        assert items==true_index


def test_find_indices_tokenized_term_in_tokenized_text():
    
    sentences=[ 
    'test sentence swaption - shops some more swaption - shops test swaption - shopsswaption - shops swaption - shops',
    'test sentence swaption - shops some more swaption - shops test swaption - shopsswaption - shops swaption - shops',    
    ''
    ]
    
    tokenized_terms=[
    'swaption - shops',
    '" swaption - shops  )',
    ''
    ]
    
    true_indices=[
    [ ( 'swaption - shops' , 14, 30), ( 'swaption - shops', 41, 57 ), ( 'swaption - shops', 96, 112 ) ],
    [ ( 'swaption - shops' , 14, 30), ( 'swaption - shops', 41, 57 ), ( 'swaption - shops', 96, 112 ) ],
    [] 
    ]
    
    for sentence, tokenized_term, true_index in zip( sentences, tokenized_terms, true_indices):
        
        items=find_indices_tokenized_term_in_text( tokenized_term , sentence )

        items=[ (item.group(0), item.span()[0], item.span()[1]) for item in items]

        assert items==true_index
        