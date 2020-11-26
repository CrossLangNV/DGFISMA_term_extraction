import pytest

from dgconcepts.pipeline.terms_defined.terms_defined_bio_tagging import get_terms_pos_bio_tags, find_indices_tokenized_term_in_text, find_defined_term_bio_tag, join_bpe

def test_join_bpe():
    
    '''
    Test the function "join_bpe"
    '''
    
    tokenized_sentences=[  
    "Test swap ##tion ' means",
    "Test swap ##tion ' means",
    "Test swap ##tion ' means",
    "Test swap ##tion ' means swap ##tion",
    "Test swap ##tion ' means swap ##tion",
    "##tion test",
    ""
    ]
    
    bio_tags=[
    "O B O O O",
    "O B I O O",
    "O O B O O",
    "O O B O O B I", 
    "O O B O O O I",
    "O O",
    ""
    ]
    
    joined_sentences_bio_tags=[
    (['Test', 'swaption', "'", 'means'], ['O', 'B', 'O', 'O']),
    (['Test', 'swaption', "'", 'means'], ['O', 'B', 'O', 'O']),
    (['Test', 'swaption', "'", 'means'], ['O', 'O', 'O', 'O']),
    (['Test', 'swaption', "'", 'means', 'swaption'], ['O', 'O', 'O', 'O', 'B']),
    (['Test', 'swaption', "'", 'means', 'swaption'], ['O', 'O', 'O', 'O', 'O']), #joining removes B, I tags from ##tokens if not preceded by B, I tag->this is what we want
    (['##tion', 'test'], ['O', 'O']),
    ([],[]) 
    ]
    
    for tokenized_sentence, bio_tag, joined_sentence_bio_tag in zip( tokenized_sentences, bio_tags, joined_sentences_bio_tags ):
        assert  joined_sentence_bio_tag  == join_bpe( tokenized_sentence.split(), bio_tag.split()   )
       
            
def test_get_terms_pos_bio_tags():
    
    '''
    Test the function "get_terms_pos_bio_tags"
    '''
    
    tokenized_sentences=[
    [ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ],
    [ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ],
    [ 'start', 'some', 'sentence', 'position', 'start', 'end', 'f', 'g', 'h---', 'start', 'end' ]
    ]
    
    bio_tags=[
        [ 'B' , 'O', 'O', 'B', 'B', 'I', 'O', 'O', 'O', 'B', 'B' ],
        [ 'O' , 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O' ],
        [ 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O' ]
    ]
    
    true_indices=[
        [ ('start', 0, 5), ('position start end', 20, 38), ('start end', 48, 57)  ],
        [ ('start', 29, 34) ],
        []
    ]
    
    for tokenized_sentence, bio_tag, true_index in zip( tokenized_sentences, bio_tags, true_indices  ):
    
        items = list( get_terms_pos_bio_tags( tokenized_sentence, bio_tag ))
    
        assert items == true_index
        
    
def test_find_indices_tokenized_term_in_nontokenized_text():
    
    '''
    Test the function "find_indices_tokenized_term_in_text", but for non-tokenized sentences.
    '''
    
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
    
    '''
    Test the function "find_indices_tokenized_term_in_text", but for tokenized sentences.
    '''
    
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
        
def test_find_defined_term_bio_tag():
    
    '''
    Test the function "find_defined_term_bio_tag".
    '''
    
    sentences=[ 
    "For the purpose of this regulation, Swaption-shops means a contract that gives the owner the right, but not the obligation, to enter a swap at or up to a certain future date or exercise date. Swaption-ships. Swaption-shops",
    "test sentence. some-()term. some-() term. some -(termm ",
    "test sentence'-. some-()term. some-() term. some -(termm ",
    "test sentence'-. Some-()term. some-() term. Some -(termm ",
    "test sentence'-. Some-()term. some-() term. Some -(termm ",
    "test sentence'-. Some-()term. some-() term. Some -(termm ",
    "Some-()term. some-() term. Some -(termm",
    ""
    ]
    
    tokenized_sentences=[
    "for the purpose of this regulation , swaption - shops means a contract that gives the owner the right , but not the obligation , to enter a swap at or up to a certain future date or exercise date . swaption - ships . swaption - shops",
    "test sentence . some - ( ) term . some - ( ) term . some - ( termm",
    "test sentence ' - . some - ( ) term . some - ( ) term . some - ( termm",
    "test sentence ' - . some - ( ) term . some - ( ) term . some - ( termm",
    "test sentence ' - . some - ( ) term . some - ( ) term . some - ( termm",
    "test sentence ' - . some - ( ) term . some - ( ) term . some - ( termm",
    "some - ( ) term . some - ( ) term . some - ( termm",
    ""
    ]
    
    tokenized_bio_tags=[
    'O O O O O O O B I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I I O O O O',
    'O O O B I I I I O B I I I I O O O O O',
    'O O O O O B I I I I O B I I I I O O O O O',
    'O O O O O B I I I O O B I I I I O O O O O', #we strip punctuation from the detected term
    'O O O O O I I B I O O B I I I I O O O O O', ##B and I following each other ==> considered same term.
    'O O O O O O I B I O O B I I I I O O O O O' ,
    'B I I I I O O O O O O O O O O O',
    ''
    ]
    
    true_indices=[ #term + indices of the bi tagged, in the original (non-tokenized) sentence
        [ ( 'Swaption-shops' , 36, 50), ( 'Swaption-ships', 192, 206 ) ],
        [ ( 'some-()term' , 15, 26), ( 'some-() term', 28, 40 ) ],
        [ ( 'some-()term' , 17, 28), ( 'some-() term', 30, 42 ) ],
        [ ( 'Some' , 17, 21), ( 'some-() term', 30, 42 ) ],
        [ ( 'Some' , 17, 21), ( 'some-() term', 30, 42 ) ],
        [ ( 'some-() term' , 30, 42)],
        [ ( 'Some-()term' , 0, 11)],
        []
    ]
    
    for sentence, tokenized_sentence, tokenized_bio_tag, true_index in zip( sentences, tokenized_sentences, tokenized_bio_tags, true_indices ):

        items=find_defined_term_bio_tag( sentence, tokenized_sentence, tokenized_bio_tag.split()  )

        items=[ (item.group(0), item.span()[0], item.span()[1]) for item in items]
        
        assert items==true_index

