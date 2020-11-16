import os
import pickle
import time

from pathlib import Path 

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences

from dgconcepts.pipeline.terms_defined_bio_tagging import bert_bio_tagging

import plac

@plac.annotations(
    #input-output
    path_test_sentences=( "Path to the test sentences (non-tokenized).", ),
    path_model_dir=( "Path to the directory where fine-tuned BERT model is stored", ),
    path_output_file=( "Path to the output file (csv)", ),
    delimiter=( "Delimiter used in output csv file", "option" ),
    #training parameters
    batch_size=( "Batch size used during inference.", "option" ),
    seq_length=( "Cut off value for number of tokens in (Bert)tokenized sentence. Bert supports sequences of up to 512 tokens." ),    
    gpu=( "GPU id. If set to -1, inference will be on cpu.", "option" )
)
def main( path_test_sentences:Path,\
          path_model_dir:Path,\
          path_output_file:Path,\
          delimiter:str="˳",\
          batch_size:int=32,\
          seq_length:int=75,\
          gpu:int=-1):
    '''
    Inference using a finetuned BertForTokenClassification model.
    '''
    
    os.makedirs( os.path.dirname( path_output_file ) , exist_ok=True)
    
    #Load the sentences on which to apply BERT_NER:
    sentences=open( path_test_sentences  ).read().rstrip("\n").split( "\n" )
    
    tokenized_texts, predictions_tags_no_pad = bert_bio_tagging( sentences, path_model_dir, gpu, seq_length=seq_length, batch_size=batch_size  )
    
    with open( path_output_file , "w"  ) as f:
        f.write( "Sentence #˳Word˳POS˳Tag\n")
        for i, (sentence, bio_tags) in enumerate( zip( tokenized_texts, predictions_tags_no_pad )):
            assert( len( sentence[:seq_length] ) == len( bio_tags ) )
            for j, (token, bio_tag ) in enumerate( zip( sentence, bio_tags ) ): 
                if j==0:
                    f.write( f"Sentence: {i+1}{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
                else:
                    f.write( f"{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
