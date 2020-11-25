import re
import os

from transformers import BertTokenizer

from pathlib import Path
import plac

def remove_quotations_around_terms(sentence:str, tag_begin="★", tag_end="☆" ) -> str:

    '''
    Helper function to remove quotation marks from annotated terms. Otherwise bio-tagger would only memorize these quotation marks.
    '''
    
    sentence=re.sub( f"{tag_begin} \‘|{tag_begin} \"|{tag_begin} \`|{tag_begin} \'|{tag_begin} \’|{tag_begin} \•|{tag_begin} \“|{tag_begin} \‧|{tag_begin} \[UNK\]" , f'{tag_begin}' ,  sentence  )
    sentence=re.sub( f"\‘ {tag_end}|\" {tag_end}|\` {tag_end}|\' {tag_end}|\’ {tag_end}|\• {tag_end}|\“ {tag_end}|\‧ {tag_end}|\[UNK\] {tag_end}" , f'{tag_end}' ,  sentence  )
    
    return sentence

def get_bio_tags(  tokenized_sentence, tag_begin="★", tag_end="☆" ):

    bio_tags=[]
    tokens=[]
    inside_term=False
    B_token=False

    for token in tokenized_sentence.split( ):

        if token==tag_begin:
            inside_term=True
            B_token=True
            continue

        elif token ==tag_end:
            inside_term=False
            B_token=False
            continue

        if inside_term and B_token:
            bio_tags.append( "B" )
            tokens.append( token )
            B_token=False

        elif inside_term and not B_token:
            bio_tags.append( "I" )
            tokens.append( token )

        else:
            bio_tags.append( "O" )
            tokens.append( token )
        
    return bio_tags, tokens

@plac.annotations(
    #input-output
    path_annotated_data=( "Path to the annotated data", ),
    path_output_file=( "Path to the output file (csv)", ),
    tag_begin=( "Tag (utf-8 character) used for annotating begin of defined term. Make sure this character is in the bert vocabulary", "option" ),
    tag_end=( "Tag (utf-8 character) used for annotating end of defined term. Make sure this character is in the bert vocabulary", "option" ),
    delimiter=( "Delimiter used in output csv file", "option" ),
    seq_length=( "Cut off value for number of tokens in (bert)tokenized sentence. Bert supports sequences of up to 512 tokens." )
)
def main( path_annotated_data:Path,\
          path_output_file:Path,\
          tag_begin:str="★", \
          tag_end:str="☆" ,\
          delimiter:str="˳",\
          seq_length:int=75 ):
    '''
    Prepare training data bio tagger.
    '''
    
    os.makedirs( os.path.dirname( path_output_file ) , exist_ok=True)
    
    definitions=open( path_annotated_data ).read().rstrip("\n").split( "\n" )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_definitions=[]
    for definition in definitions:
        tokenized_definitions.append(" ".join(tokenizer.tokenize( definition ) ) )

    tokenized_definitions=[ remove_quotations_around_terms( sentence ) for sentence in tokenized_definitions if sentence.strip() ]

    #print the data to csv:

    sentences_bio_tags=[]
    sentences=[]
    for sentence in tokenized_definitions:
        bio_tags, tokens=get_bio_tags( sentence )
        sentences_bio_tags.append(  bio_tags)
        sentences.append( tokens )

    assert( len(sentences) ==len( sentences_bio_tags ) )
    
    with open( path_output_file , "w"  ) as f:
        f.write( "Sentence #˳Word˳POS˳Tag\n")
        for i, (sentence, bio_tags) in enumerate( zip( sentences, sentences_bio_tags )):
            assert( len( sentence ) == len( bio_tags ) )
            for j, (token, bio_tag ) in enumerate( zip( sentence[:seq_length], bio_tags ) ): 
                if j==0:
                    f.write( f"Sentence: {i+1}{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
                else:
                    f.write( f"{delimiter}{token}{delimiter}--{delimiter}{bio_tag}\n")
    
if __name__ == "__main__":
    plac.call(main)
