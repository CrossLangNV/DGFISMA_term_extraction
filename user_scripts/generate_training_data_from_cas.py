from typing import List, Tuple

def annotations_to_tags( sentence:str, defined:List[ Tuple[ int , int ] ] , tag_begin="★", tag_end="☆"  ):

    '''
    Given a sentence, and a list of Tuples representing annnotations, this function converts the sentence to a sentence with annotations that can be used as input to the generate_training_data.py user script.
    '''
    
    sentence_new=''
        
    for i in range( len(defined) ):
        if i==0:
            sentence_new=sentence[ :defined[i][0] ]
        sentence_new=sentence_new + f" {tag_begin} " + sentence[ defined[i][0]:defined[i][1] ] + f" {tag_end} "
        if i<len(defined)-1:
            sentence_new=sentence_new + sentence[ defined[i][1] : defined[i+1][0] ]
        else:
            sentence_new=sentence_new + sentence[ defined[i][1] :]
        
    return sentence_new