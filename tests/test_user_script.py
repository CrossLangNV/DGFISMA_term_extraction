from user_scripts.generate_training_data_from_cas import annotations_to_tags

def test_annotations_to_tags():
    sentences=[
        'This is a test sentence'
    ]
    
    list_defined=[
    [ ( 0,4 ), ( 10,14 ) ],
    [ ( 0,4 ), (8,9), ( 10,14 ) ],
    [ ( 0,4 ) ]
    ]
    
    sentences_new=[
        ' ★ This ☆  is a  ★ test ☆  sentence',
        ' ★ This ☆  is  ★ a ☆   ★ test ☆  sentence',
        ' ★ This ☆  is a test sentence'
    ]
    
    for sentence, defined, sentence_new, in zip( sentences, list_defined, sentences_new ):
        assert annotations_to_tags( sentence, defined ) == sentence_new
        