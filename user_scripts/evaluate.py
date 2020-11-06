import pandas as pd
from pathlib import Path 
import plac

from seqeval.metrics import classification_report

from .utils import SentenceGetter

def get_sentences_labels( path: str, delimiter:str="˳" ):
    
    data=pd.read_csv( path, delimiter=delimiter, engine='python' )
    data=data.fillna( method='ffill'  )
    
    getter=SentenceGetter(data)

    labels = [[s[2] for s in sentence] for sentence in getter.sentences]
    
    return labels

@plac.annotations(
    #input-output
    path_predicted_labels=( "Path to the predicted labels (in csv format)", ),
    path_true_labels=( "Path to the gold standard labels (in csv format)", ),
    delimiter=( "delimiter used in csv files", "option" )
)
def main( path_predicted_labels:Path,\
          path_true_labels:Path, \
          delimiter:str="˳" ):
    '''
    Evaluate results bio tagger.
    '''
    
    predicted_labels=get_sentences_labels( path_predicted_labels, delimiter=delimiter )
    true_labels=get_sentences_labels( path_true_labels, delimiter)
    
    scores=classification_report( predicted_labels, true_labels  )
    
    print( scores )
    
    return scores

if __name__ == "__main__":
    plac.call(main)
