# term-extraction

## Instructions

use "dbuild.sh" to build the docker image

use "dcli.sh" to start a docker container

Given a json, e.g.: (e.g. https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/tests/files/test_files/jsons_with_definitions/doc_3b30d182-e395-5e6c-991a-b57cd01598d0_test_response.json) , with a "cas_content" and "content_type" field, a json with the same fields will be returned (e.g. https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/tests/files/test_files/jsons_with_definitions_terms/doc_3b30d182-e395-5e6c-991a-b57cd01598d0.json) , but with extra annotations added (see next sections).

The "cas_content" is a UIMA CAS object, encoded in base64. The "content_type" can be "html" or "pdf". 

For working with CAS object, [the dkpro-cassis library](https://github.com/dkpro/dkpro-cassis) is used.

A pretrained BertBIOTagger (see below), and training data is provided https://github.com/CrossLangNV/DGFISMA_term_extraction/releases/tag/0.0.1.

### Configuration

We provide a configuration file: https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/media/TermExtraction.config
```
[TermExtraction]
SPACY_MODEL=en_core_web_lg
N_JOBS=1
BATCH_SIZE=32
MAX_LEN_NGRAM = 4
EXTRACT_SUPERGRAMS=False
TFIDF_REGEX=-2.0
TFIDF_WHITELIST=-1.0
TFIDF_BERT=-3.0

[BertBIOTagger]
PATH_MODEL_DIR=Fine_tuned_models/dgfisma_bio_tag
GPU=-1
NUM_THREADS_CPU=-1
SEQ_LENGTH=75
BATCH_SIZE=32

[DefinedTerm]
BERT_BIO_TAGGING=True
USE_REGEX=False
USE_WHITELIST=True
USE_TF_IDF=True
USE_BERT=True
FALLBACK_TO_REGEX=False
FALLBACK_TO_WHITELIST=False
FALLBACK_TO_TF_IDF=False

[Regex]
REGEX_TERMS = [\‘\"\`\'\•\“\‧][a-z0-9\-(){}_/\\:’]{2,}[a-z0-9 \-(){}_/\\:’]*[a-z0-9\-(){}_/\\:’]+[\‘\"\`\'\•\”\‧’]\B
REGEX_ABBREVIATIONS = [\‘\"\`\'\•\“\‧][A-Z’]{2,}[\‘\"\`\'\•\”\‧\’]\B

[Annotation]
SOFA_ID=html2textView
VALUE_BETWEEN_TAG_TYPE=com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType
TAG_NAMES=p
TOKEN_TYPE=de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf
LEMMA_TYPE=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma
DEPENDENCY_TYPE=de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency
DEFINITION_TYPE=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
DEFINED_TYPE=cassis.Token
PARAGRAPH_TYPE=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph

[Annotation_user]
DEFINITION_TYPE_USER=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence_user
DEFINED_TYPE_USER=cassis.Token_user
TOKEN_TYPE_USER=de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf_user
```

This allows configuration of the term extraction pipeline.

#### TermExtraction

The Term Extraction algorithm consists of the following steps:

1. Retrieve text segments:
    
    We select the view  `config[ 'Annotation' ].get( 'SOFA_ID' )` of the CAS object, and retrieve all `config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )` features with tagname equal to `config[ 'Annotation' ].get( 'TAG_NAMES' )`. This results in a list of sentences (list of strings).
    
2. Extract terms from text segments :
   
    Next, the Spacy model `config[ 'Annotation' ].get( 'SPACY_MODEL' )` will process these sentences and return a Doc object with various annotations.
   Doc.noun_chunks is a generator (list) containing all 'flat' noun phrases, i.e. phrases with a noun as their head.
   POS tags, as well as a list of invalid words, are used to filter the terms.
   Dependency values of nouns in the Doc are used for the extraction of complex noun phrases, i.e. supergrams. 
   With the head noun as root, its attributes left_edge and right_edge will provide the first and the last tokens of the subtrees.
   Supergrams are therefore Doc spans from the root's left_edge's index up to and including its right_edge. If `config[ 'Annotation' ].get( 'EXTRACT_SUPERGRAMS' )` is set to False, no supergrams will be obtained in this step. 

3. Calculate tf-idf score for the extracted terms

   [Scikit-learn's](https://scikit-learn.org/stable/) Tfidftransformer is used to calculate the tf-idf score for each term in the document. 

4. Rule based retrieval of terms:

    Retrieve detected definitions, annotated as a `config[ 'Annotation' ].get( 'DEFINITION_TYPE' )` feature, and search for text in between 'quotes' using a regex. This allows us to retrieve terms that are difficult to find using classic NLP methods. The tf-idf score of this set of terms is set to `config[ 'Annotation' ].get( 'TF_IDF_REGEX' )`. The regex to find text in between quotes can be configured via the `config[ 'Regex' ].get( 'REGEX_TERMS' )` and `config[ 'Regex' ].get( 'REGEX_ABBREVIATIONS' )` fields. Matches of both regexes are considered terms and will be given the tf-idf score `config[ 'Annotation' ].get( 'TF_IDF_REGEX' )`. 
    
    Important note: `config[ 'Annotation' ].get( 'DEFINITION_TYPE' )` annotations should be added to the cas using the API for definition detection, see: https://github.com/CrossLangNV/DGFISMA_definition_extraction.
    
5. Whitelist + blacklist:

    In `media/whitelist.txt` and `media/blacklist.txt`, users can provide a blacklist and a whitelist of terms. Blacklisted terms will be removed from the term list found in step 3, and whitelisted terms will be added to the list. The tf-idf score of whitelisted terms is set to `config[ 'Annotation' ].get( 'TF_IDF_WHITELIST' )`. The tf-idf score of terms found in step 4 that are also in the whitelist remains `config[ 'Annotation' ].get( 'TF_IDF_REGEX' )`.
    
5. Annotation of terms:

    Using the [Aho-Corasick algorithm](https://pypi.org/project/pyahocorasick/), the offsets of the terms found via step 2 to 5 are calculated. These offsets are then used to add the `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` annotation to the cas. This feature type contains both the offsets, the lowercased version of the term and the tf-idf score ( <em> .begin, .end, .term, .tfidfValue </em> ).
    
6. Annotation of lemmata:

    Lemmata of terms are obtained using the Spacy model `config[ 'Annotation' ].get( 'SPACY_MODEL' )`. Lemmata annotations are added at the obtained offset of the original term using the `config[ 'Annotation' ].get( 'LEMMA_TYPE' )` feature type  ( <em> .begin, .end, .value </em> ). 
    
`config[ 'TermExtraction' ].get( 'N_JOBS' )` and `config[ 'TermExtraction' ].get( 'BATCH_SIZE' )` allows for configuration of the number of processes and batch size used by the Spacy model.

    
#### BertBIOTagger

We refer to https://github.com/CrossLangNV/DGFISMA_term_extraction/tree/master/user_scripts for more information on training of a BertForTokenClassification model for BIO tagging. In short, the BertBIOTagger is used to detect the defined term in a definition. 

The `config[ 'Annotation' ].get( 'PATH_MODEL_DIR' )` is the path to the trained BertBIOTagger relative to the MODELS folder that should be created in the directory where this repository is cloned. See https://github.com/CrossLangNV/DGFISMA_term_extraction/releases/tag/0.0.1 for such a trained BertBIOTagger.

It is recommended to set `config[ 'Annotation' ].get( 'SEQ_LENGTH' )` (sequence length used during inference) to the same number as used during training of the model. 

If a set of tokens in a definition (i.e. covered by a `config[ 'Annotation' ].get( 'DEFINITION_TYPE' )` annotation)  are labeled with B or I tags, they are annotated with the `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` feature.

Use of the BertBIOTagger is optional. If `config[ 'DefinedTerm' ].get( 'BERT_BIO_TAGGING' )` is set to False, the `config[ 'Annotation' ].get( 'SPACY_MODEL' )` will be used for dependency parsing.

If `config[ 'BertBIOTagger' ].get( 'GPU' )` is set to -1, CPU will be used for inference. `config[ 'BertBIOTagger' ].get( 'NUM_THREADS_CPU' )` allows for configuration of the number of threads used during inference on CPU. If `config[ 'BertBIOTagger' ].get( 'GPU' )` > -1, this will be ignored. When `config[ 'BertBIOTagger' ].get( 'NUM_THREADS_CPU' )` is set to -1, all available threads will be used.

##### Example:

Given the definitions:

`Asset management means something.` </br>
`The Profit estimate is defined as something else.`

And the following tags predicted by the BertBIOTagger:

Sentence #  | Word | Tag  
--- | --- | --- |
Sentence 1 |  asset | B 
 / |  management | I 
 / |  means | O 
 / |  something | O 
 / |  . | O 
 Sentence 2 |  the | O  
 /|  profit | I 
 / |  estimate | B 
 /|  is | O 
 /|  defined | O 
 /|  as | O 
 /|  something | O 
/|  else | O 
/|  . | O

Then the term `Asset management` with offset 0 - 16 , and the term `Profit estimate` with offset 4 - 19, will be annotated with `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )`. 

Note that for robustness the tags I, B are not treated differently. E.g. O, B, I, I, O and O, I, I, B, O both indicate a defined term consisting of 3 tokens starting at position 1 and ending at position 3.

#### DefinedTerm

Defined Terms (i.e. terms that are defined by a definition, annotated as `config[ 'Annotation' ].get( 'DEFINITION_TYPE' )` ) are annotated as `config[ 'Annotation' ].get( 'DEFINED_TYPE' )`.

If precision/recall of the BertBIOTagger is sufficiently high, one can use the following configuration:

```
[DefinedTerm]
BERT_BIO_TAGGING=True
USE_REGEX=False
USE_WHITELIST=False
USE_TF_IDF=False
USE_BERT=True
FALLBACK_TO_REGEX=False
FALLBACK_TO_WHITELIST=False
FALLBACK_TO_TF_IDF=False
```

This configuration will assume all `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotations are correct, and indicate terms that are defined by the definition. At the offset of the `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotation, a `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` annotation will simply be added. The term at `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` is then annotated as a `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` in the document, with `config[ 'TermExtraction' ].get( 'TFIDF_BERT' )`, if it was not already detected by the TermExtraction algorithm. 

However, to increase precision, we recommend using the following configuration:

```
[DefinedTerm]
BERT_BIO_TAGGING=True
USE_REGEX=False
USE_WHITELIST=True
USE_TF_IDF=True
USE_BERT=True
FALLBACK_TO_REGEX=False
FALLBACK_TO_WHITELIST=False
FALLBACK_TO_TF_IDF=False
```

Using this configuration, terms that are whitelisted (+terms detected in definitions between quotes (i.e. via a regex), will be considered the Defined term if they have an overlap with a `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotation. If such a term is not found, the algorithm will consider other terms annotated as `config[ 'Annotation' ].get( 'TOKEN_TYPE' )`, due to USE_TF_IDF being set to True (see below).

A small example will make this clear. Consider the following definition:

`The 'Profit estimate' is defined as something else.`

If our BertBIOTagger mistakenly only tags `Profit` with a B or I tag, we would label `Profit` as the defined term. However, by looking at `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` annotations sharing an intersection with the `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotation, we can correct `Profit` to `Profit estimate`. 

If USE_REGEX is set to True, terms in between 'quotes' (detected via a regex), will be considered the first candidates for obtaining a `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` annotation. If such a defined term is found in a definition, the algorithm will stop searching, and proceed to the following detected definition. If no defined term is found, the algorithm will consider the other candidates.

If USE_WHITELIST is set to True, terms in the whitelist and in between 'quotes' will be considered candidates. (Terms in between quotes are considered whitelisted).

If USE_TF_IDF is set to True, other terms found via the TermExtraction algorithm will be considered. 

Setting USE_BERT to True, means that, if no `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` is found by looking at overlap between `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotations and `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` annotation (given one of the USE_REGEX, USE_WHITELIST, USE_TF_IDF flags are set to True), the `config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )` annotation offset is added as a `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` annotation, as explained above.

Setting FALLBACK_TO_REGEX, FALL_BACK_TO_WHITELIST or FALL_BACK_TO_TF_IDF to True means that all terms in between quotes, whitelisted terms, or `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` annotations are annotated as defined terms (i.e. as `config[ 'Annotation' ].get( 'DEFINED_TYPE' )` ). This is not recommended. 

#### User annotations

The annotations `config[ 'Annotation_user' ].get( 'TOKEN_TYPE_USER' )`, `config[ 'Annotation' ].get( 'DEFINED_TYPE_USER' )` and `config[ 'Annotation' ].get( 'DEFINITION_TYPE_USER' )` are annotations added by the users of the Glossary app. The provided user scripts can be used to generate whitelisted terms, training data for the BertBIOTagger for detection of defined terms and the [DistilBertSequenceClassifier](https://github.com/CrossLangNV/DGFISMA_definition_extraction) for detection of definitions.

## Testing:

Unit tests can be run with pytest:

`pytest`