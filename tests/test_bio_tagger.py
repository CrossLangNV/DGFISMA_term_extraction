import pytest

from dgconcepts.pipeline.definitions_bio_tagging import get_terms_pos_bio_tags



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