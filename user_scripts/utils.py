import re
from pandas.core.frame import DataFrame

class SentenceGetter(object):

    def __init__(self, data: DataFrame):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
def remove_quotations_around_terms(sentence:str, tag_begin="★", tag_end="☆" ) -> str:

    '''
    Helper function to remove quotation marks from annotated terms. Otherwise bio-tagger would only memorize these quotation marks.
    '''
    
    sentence=re.sub(  f"({tag_begin} *[\‘\"\`\'\’\•\“\‧\[UNK\]])", f" {tag_begin} ", sentence )
    sentence=re.sub(  f"([\‘\"\`\'\’\•\“\‧\[UNK\]] *{tag_begin})", f" {tag_begin} ", sentence )

    sentence=re.sub(  f"({tag_end} *[\‘\"\`\'\’\•\“\‧\[UNK\]])", f" {tag_end} ", sentence )
    sentence=re.sub(  f"([\‘\"\`\'\’\•\“\‧\[UNK\]] *{tag_end})", f" {tag_end} ", sentence )

    return sentence