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
    
    with open( path_output_file , "w"  ) as f:
        f.write( "Sentence #˳Word˳POS˳Tag\n")
        for i, (sentence, bio_tags) in enumerate( zip( tokenized_texts, predictions_tags_no_pad )):
            assert( len( sentence[:seq_length] ) == len( bio_tags ) )
            for j, (token, bio_tag ) in enumerate( zip( sentence, bio_tags ) ): 
                if j==0:
                    f.write( f"Sentence: {i+1}{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
                else:
                    f.write( f"{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
