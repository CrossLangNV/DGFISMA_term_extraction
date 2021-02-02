import os
import pickle
import time
import string
import re

from typing import List, Tuple, Generator

from pathlib import Path 

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences

from ..utils import is_token

class TrainedBertBIOTagger( ):
    
    '''
    A trained BertForTokenClassification model.
    '''
    
    def __init__( self,  path_model_dir: Path ):
        
        '''
        :param path_model_dir: Path. Path to the trained BertForTokenClassification model.
        '''

        self._path_model_dir=path_model_dir
        
    def load_model(self ):
        
        '''
        Load a trained BertForTokenClassification models, and accompanying BertTokenizer. The BertTokenizer will lowercase the tokens. If this needs to be changed, please also change it in the user_scripts.train module.
        '''
        
        with open( os.path.join( self._path_model_dir , "tags_vals" ) , "rb") as fp:
            self.tags_vals = pickle.load(fp)
         
        #sanity check
        assert set( [ 'B', 'I', 'O', 'PAD' ] ) == set(self.tags_vals)
        
        self.model = BertForTokenClassification.from_pretrained( self._path_model_dir , num_labels=len(self.tags_vals) )
        self.tokenizer=BertTokenizer.from_pretrained( self._path_model_dir, do_lower_case=True )

        
def process_definitions_bert_bio_tagging( sentences: List[str], trained_bert_bio_tagger: TrainedBertBIOTagger, gpu:int=-1, num_threads:int=-1,\
                                         seq_length:int=75, batch_size:int=32 ) -> Generator[ List[ Tuple[ str, int, int ] ], None, None ]:
    
    '''
    Function will use TrainedBertBIOTagger model to tokenize the sentences and BIO tag the sentences, using the bert_bio_tagging function. 
    Next the BIO tags are converted to offsets in the original (non-tokenized) sentence using the join_bpe and find_defined_term_bio_tag functions. 
    Function returns a Generator yielding a list of terms (defined_terms) and offsets for each sentence.
    
    :param sentences: List. List of strings. 
    :param trained_bert_bio_tagger: TrainedBertBIOTagger. 
    :param gpu: int. GPU id. 
    :param num_threads: int. Nr of CPU threads used during inference. Ignored when GPU>-1.
    :param seq_length: int. Sequence length. 
    :param batch_size: int.
    :return: Generator.
    '''
    
    #inference
    tokenized_sentences, bio_tags = bert_bio_tagging( sentences, trained_bert_bio_tagger, gpu, num_threads, seq_length=seq_length, batch_size=batch_size  )

    #sanity check
    assert( len( sentences ) == len( tokenized_sentences ) == len( bio_tags ) )
    
    joint_tokenized_sentences=[]
    joint_tokenized_bio_tags=[]

    #join tokenized sentences  (i.e. remove the ##) and take care of the bio tags
    for tokenized_sentence, bio_tag in zip( tokenized_sentences , bio_tags ):
        joint_tokenized_sentence, joint_tokenized_bio_tag=join_bpe( tokenized_sentence, bio_tag )
        joint_tokenized_sentences.append( joint_tokenized_sentence )
        joint_tokenized_bio_tags.append( joint_tokenized_bio_tag )

    #sanity check
    assert( len( sentences ) == len( joint_tokenized_sentences  ) == len(  joint_tokenized_bio_tags  )  )
    
    #find the offset in the original sentence
    for i in range( len( sentences ) ):
        detected_terms=find_defined_term_bio_tag( sentences[i], " ".join(joint_tokenized_sentences[i]) , joint_tokenized_bio_tags[i]  )
        if not detected_terms:
            yield []
            continue
        yield [ (detected_term.group(0),  detected_term.span()[0], detected_term.span()[1]) for detected_term in detected_terms ] 


def bert_bio_tagging( sentences: List[str], trained_bert_bio_tagger: TrainedBertBIOTagger, gpu:int=-1, num_threads:int=-1, \
                     seq_length:int=75, batch_size:int=32) -> Tuple[ List[str], List[str] ]:

    '''
    Inference using TrainedBertBIOTagger. Sentences will be padded to seq_length before being send to TrainedBertBIOTagger in batches. 
    Returns tokenized texts (original length of input sentences) and BIO tags for each token (not containing predictions for padded elements/or stripped tokens due seq_length). Length of the predictions (BIO) tags may thus differ from the length of the tokenized_texts.
    
    :param sentences: List. List of strings.
    :param trained_bert_bio_tagger: TrainedBertBIOTagger. 
    :param gpu: int. GPU id. 
    :param num_threads: int. Nr of CPU threads used during inference. Ignored when GPU>-1.
    :param seq_length: int. Sequence length. 
    :param batch_size: int.
    :return: Tuple. Returns a list of tokenized sentences and predicted tags. 
    '''
            
    tags_vals=trained_bert_bio_tagger.tags_vals

    #Put the model on the GPU
    if torch.cuda.is_available() and gpu > -1:
        trained_bert_bio_tagger.model.cuda(gpu) 
        print( f"Inference (Bert BIO tagging) on gpu {gpu}." )
    else:
        if num_threads==-1:  
            #Use all available threads. By default all available threads are used by pytorch.
            num_threads=torch.get_num_threads()
        else:
            torch.set_num_threads( num_threads )
        print( f"Inference (Bert BIO tagging) on cpu using {num_threads} thread(s)." )

    trained_bert_bio_tagger.model.eval()

    #tokenize the sentences:
    tokenized_texts = [ trained_bert_bio_tagger.tokenizer.tokenize(sent) for sent in sentences]

    #padding of sentences:
    test_input_ids = pad_sequences([trained_bert_bio_tagger.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
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
            logits = trained_bert_bio_tagger.model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits[0].detach().cpu().numpy()

        predictions_batch = np.argmax(logits, axis=2)  #get the predicted labels

        for prediction_batch in predictions_batch:
            predictions.append(prediction_batch)
            pred_batch_tags=[tags_vals[ p ] for p in prediction_batch ]
            predictions_tags.append( pred_batch_tags )

    end=time.time()

    total_number_of_words=len([j for sub in tokenized_texts for j in sub])

    print( f"Bert BIO tagging on {len(sentences)} sentences took {end-start} seconds (  { total_number_of_words/(end-start) } words/s ) "  )

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
    Merge bpe tokens of a bert tokenized sentence and process bio_tags. Tokens with no bio tag (i.e. at position>seq_length) are ignored (stripped).
    
    :param tokens: List. List of strings (bpe tokens).
    :param tags: List. List of strings (predicted tag for each bpe token)
    :return: Tuple. Returns a list of bpe-merged tokens and predicted tags.     
    '''

    new_tokens, new_tags = [], []

    #tokens at position>seq_length are ignored
    for token, tag in zip(tokens, tags ):
        if token.startswith("##") and new_tokens: #check for new_tokens, to deal with case if sentence should start with ##
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_tags.append(tag)
            new_tokens.append(token)
    return new_tokens, new_tags


def get_terms_pos_bio_tags( tokens:List[str], bio_tags:List[str] ) -> Generator[ Tuple[ str, int, int  ], None, None ]:
    
    '''
    Function to get the tagged term + span found via TrainedBertBIOTagger. Note that the tags I, B are not treated differently, for robustness. E.g. O, B, I, I, O and O, I, I, B, O both indicate a term consisting of 3 tokens starting at position 1 and ending at position 3.
    Function returns a Generator yielding a tuple containing the tagged term (i.e. tagged with B, I tags) and offset in detokenized sentence.
    
    :param tokens: List.
    :param bio_tags: List.
    :return: Generator.
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
        
        
def find_indices_tokenized_term_in_text( tokenized_term: str, sentence: str   ) -> Generator:

    '''
    Find matches of a bert tokenized term (after merging of bpe tokens with function join_bpe) in a tokenized (after merging of bpe-tokens)/non tokenized sentence using regex.
    Leading and trailing punctuation is stripped from the token (and will thus also not be in the span).
    
    :param tokenized_term: String.
    :param sentence: String.
    :return: Generator. Generator yielding a regex match.
    '''
        
    extra_punctuation_tokens='‘\"`\'’•”‧'

    if '[UNK]' in tokenized_term: #if the term contains an [UNK] ==> probably not a legit term
        return
    
    #strip leading and trailing punctuation from this term, to makes sure that we don't annotate terms consisting solely of punctuations as terms
    #+for safety and better precision, although Bert tokenization should have tokenized these punctuations: i.e. 'some' ==> ' some ' 

    tokenized_term=tokenized_term.strip( string.punctuation+extra_punctuation_tokens+" " )
    
    sentence=sentence.replace( "\xa0", " " )
        
    #make regex
    term_regex=r''
    for token in tokenized_term.split():
        if not term_regex:
            term_regex=token
        else:
            term_regex=term_regex+r"([ \n]*)"+re.escape( token )
    
    if tokenized_term:
        for match in re.finditer( term_regex , sentence , re.IGNORECASE):
            if is_token( match.span()[0], match.span()[1]-1, sentence, special_characters=[] ):
                yield match
        
        
def find_defined_term_bio_tag( sentence:str , tokenized_sentence:str, tokenized_bio_tags:List[ str ], verbose=True  ) -> Generator:
    
    '''
    Find offset of the detected term (via bio tag), in the original (non-tokenized) sentence. 
    Note that punctuation is stripped from the detected term. I.e. if b,i tags cover a punctuation, this is stripped before lookup via regex (for calculation of offsets).
    
    :param sentence: String. Original (non-tokenized) sentence.
    :param tokenized_sentence: String. Bert tokenized sentence after merging of bpe tokens (joining of ##'s).
    :param tokenized_bio_tags: List. Tags predicted by a TrainedBertBIOTagger for each token (after processing with join_bpe function).
    :return: Generator. Generator yielding regex matches of detected terms in the original sentence.
    '''
    
    #Sanity check.
    assert (len (tokenized_sentence.split()) == len( tokenized_bio_tags ))
    
    for term, span_begin, span_end in get_terms_pos_bio_tags( tokenized_sentence.split(), tokenized_bio_tags ):
        matches=[]
        for match in find_indices_tokenized_term_in_text( term, tokenized_sentence  ):

            is_defined=False

            if match.span()[0]>=span_begin and match.span()[1]<=span_end: #check if it is a defined term
                is_defined=True

            matches.append( (match, is_defined) )

        if not matches and verbose:
            print( f"Could not find detected term: '{term}' in tokenized sentence '{tokenized_sentence}'. If the term only consists of punctuation, or if the term contains [UNK]'s this is expected behaviour." )
            continue

        matches_original_sentence=list(find_indices_tokenized_term_in_text( term, sentence ))

        if (len( matches_original_sentence ) < len( matches ) ) and verbose :  #number of matches in original sentence could be greater than number of matches in tokenized sentence, due to seq_length
            print( f"Number of occurences of tokenized term: '{term}' in tokenized sentence: '{tokenized_sentence}' is greater than the number of occurences in non-tokenized sentence: '{sentence}'. This could be unwanted behaviour." )
            continue

        for match_original, match_tokenized in zip( matches_original_sentence, matches ):

            if match_tokenized[-1]==True:
                yield match_original

