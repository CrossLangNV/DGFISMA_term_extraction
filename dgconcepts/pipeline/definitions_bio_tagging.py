import os
import pickle
import time
import string
import re

from typing import List, Tuple

from pathlib import Path 

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences


from .utils import  is_token


def bert_bio_tagging( sentences: List[str], path_model_dir:Path, gpu:int=-1, seq_length:int=75, batch_size:int=32) -> Tuple[ List[str], List[str] ]:

    '''
    Bio tagging with finetuned BertForTokenClassification model.
    '''
    
    #Load the model and the data file with tags:
    with open( os.path.join( path_model_dir , "tags_vals" ) , "rb") as fp:
        tags_vals = pickle.load(fp)

    model = BertForTokenClassification.from_pretrained( path_model_dir , num_labels=len(tags_vals) )
    tokenizer = BertTokenizer.from_pretrained( path_model_dir, do_lower_case=True )

    #Put the model on the GPU
    if torch.cuda.is_available() and gpu > -1:
        model.cuda(gpu) 
        print( f"inference on gpu {gpu}" )
    else:
        print( "inference on cpu" )

    model.eval()

    #tokenize the sentences:
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    #padding of sentences:
    test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=seq_length, dtype="long", truncating="post" , padding="post")

    test_attention_masks = [[float(i>0) for i in ii] for ii in test_input_ids]

    #convert to pytorch tensors:
    test_input_ids = torch.tensor(test_input_ids)
    test_attention_masks = torch.tensor(test_attention_masks)

    #define dataloaders:
    test_data = TensorDataset( test_input_ids,  test_attention_masks )
    test_sampler = SequentialSampler(test_data)  #at test time, we pass the data sequentially
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size= batch_size )

    predictions=[]
    predictions_tags=[]

    start=time.time()

    for batch in test_dataloader:
        if torch.cuda.is_available() and gpu > -1:
            batch = tuple(t.cuda( gpu ) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits[0].detach().cpu().numpy()

        predictions_batch = np.argmax(logits, axis=2)  #get the predicted labels

        for prediction_batch in predictions_batch:
            predictions.append(prediction_batch)
            pred_batch_tags=[tags_vals[ p ] for p in prediction_batch ]
            predictions_tags.append( pred_batch_tags )

    end=time.time()

    total_number_of_words=len([j for sub in tokenized_texts for j in sub])

    print( f"NER on {len(sentences)} sentences took {end-start} seconds (  { total_number_of_words/(end-start) } words/s ) "  )

    assert len(tokenized_texts) == len(predictions_tags)

    predictions_tags_no_pad=[]
    for sentence, prediction_tags in zip(tokenized_texts, predictions_tags  ):
        prediction_tags_no_pad=[]
        for i in range( len(sentence[:seq_length] ) ):
            prediction_tags_no_pad.append(prediction_tags[i])
        predictions_tags_no_pad.append(prediction_tags_no_pad)

    return tokenized_texts, predictions_tags_no_pad


def join_bpe( tokens:List[str], tags:List[str]  )->Tuple[ List[str], List[str] ]:
    
    '''
    Detokenize bert tokenized sentence and process bio_tags. Tokens with no bio tag (i.e. at position>seq_length) are ignored.
    '''

    new_tokens, new_tags = [], []

    #tokens at position>seq_length are ignored
    for token, tag in zip(tokens, tags ):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_tags.append(tag)
            new_tokens.append(token)
    return new_tokens, new_tags


def get_terms_pos_bio_tags( tokens:List[str], bio_tags:List[str] ):
    
    '''
    Function to get the tokenized term + span found via Bert bio tagger.
    '''
    
    detected_term=[]
    
    start_index=0
    position=0
    
    for token, bio_tag in zip( tokens, bio_tags ):
                
        if bio_tag!="O":
            if not detected_term: #newly detected token
                start_index=position
            detected_term.append( token )


        elif bio_tag=="O" and detected_term:
            yield( (" ".join(detected_term), start_index, start_index+len(" ".join(detected_term))  )  )
            detected_term=[]
            detected_term_pos=[]
            
        position=position+len(token+" ")  #position in detokenized sentence

    #add the last one
    if detected_term:
        yield( (" ".join(detected_term), start_index, start_index+len(" ".join(detected_term))  )  )
        
        
def find_indices_tokenized_term_in_text( tokenized_term: str, sentence: str   ):

    '''
    Find matches of a bert tokenized term (after ##'s are joined) in a tokenized/non tokenized sentence using regex.
    Leading and trailing punctuation is stripped from the token (and will thus also not be in the span).
    '''
    
    extra_punctuation_tokens='‘\"`\'’•”‧'

    #strip leading and trailing punctuation from this term
    tokenized_term=tokenized_term.strip( string.punctuation+extra_punctuation_tokens+" " )
    
    #make regex
    term_regex=''
    for token in tokenized_term.split():
        if not term_regex:
            term_regex=token
        elif token in (string.punctuation+extra_punctuation_tokens ):
            #we have to escape special characters (i.e. . without \. would match whatever character)
            term_regex=term_regex+"([ ]*)"+"\\"+token
        else:
            term_regex=term_regex+"([ ]*)"+token
    
    if tokenized_term:
        for match in re.finditer( term_regex , sentence , re.IGNORECASE):
            if is_token( match.span()[0], match.span()[1]-1, sentence, special_characters=[] ):
                yield match
        
'''
def find_indices_tokenized_term_in_nontokenized_text(tokenized_term: str, sentence: str):
    

    #first strip possible leading or trailing punctuation from the bert tokenized term:
    #tokenized_term=tokenized_term.strip( string.punctuation+'‘\"`\'’•”‧'+" "  )
    
    extra_punctuation_tokens='‘\"`\'’•”‧'
    
    tokenized_term=tokenized_term.strip( string.punctuation+extra_punctuation_tokens+" " )
            
    #make regex
    term_regex=''
    for token in tokenized_term.split():
        if not term_regex:
            term_regex=token
        elif token in (string.punctuation+extra_punctuation_tokens ):
            #we have to escape special characters (i.e. . without \. would match whatever character)
            term_regex=term_regex+"([ ]*)"+"\\"+token
        else:
            term_regex=term_regex+"([ ]*)"+token
    
    if tokenized_term:
        for match in re.finditer(term_regex, sentence, re.IGNORECASE):
            if is_token( match.span()[0], match.span()[1]-1, sentence, special_characters=[] ):
                yield match
'''