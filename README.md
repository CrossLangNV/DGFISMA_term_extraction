# term-extraction

## Instructions

use "dbuild.sh" to build the docker image

use "dcli.sh" to start a docker container

Given a json, e.g.: (e.g. https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/tests/files/test_files/jsons_with_definitions/doc_3b30d182-e395-5e6c-991a-b57cd01598d0_test_response.json) , with a "cas_content" and "content_type" field, a json with the same fields will be returned (e.g. https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/tests/files/test_files/jsons_with_definitions_terms/doc_3b30d182-e395-5e6c-991a-b57cd01598d0.json) , but with extra annotations added (see next sections).

The "cas_content" is a UIMA CAS object, encoded in base64. The "content_type" can be "html" or "pdf". 

For working with CAS object, [the dkpro-cassis library](https://github.com/dkpro/dkpro-cassis) is used.


### Configuration

We provide a configuration file: https://github.com/CrossLangNV/DGFISMA_term_extraction/blob/master/media/TermExtraction.config
```
[TermExtraction]
SPACY_MODEL=en_core_web_lg
MAX_LEN_NGRAM = 4
EXTRACT_SUPERGRAMS=False
TFIDF_REGEX=-2.0
TFIDF_WHITELIST=-1.0
TFIDF_BERT=-3.0

[DefinedTerm]
BERT_BIO_TAGGING=True
USE_REGEX=True
USE_WHITELIST=True
USE_TF_IDF=True
USE_BERT=True
FALLBACK_TO_REGEX=False
FALLBACK_TO_WHITELIST=False
FALLBACK_TO_TF_IDF=False

[BertBIOTagger]
PATH_MODEL_DIR=Fine_tuned_models/dgfisma_bio_tag
GPU=-1
SEQ_LENGTH=75
BATCH_SIZE=32

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
   Supergrams are therefore Doc spans from the root's left_edge's index up to and including its right_edge. If `config[ 'Annotation' ].get( 'EXTRACT_SUPERGRAMS' )` is set to False, no supergrams will be extracted. 

3. Calculate tf-idf for the extracted terms

   [Scikit-learn's](https://scikit-learn.org/stable/) Tfidftransformer then is used to calculate the tf-idf score for each term in the document. 

4. Rule based retrieval of terms:

    Retrieve detected definitions, annotated as a `config[ 'Annotation' ].get( 'DEFINITION_TYPE' )` feature, and search for text in between 'quotes' using a regex. This allows us to retrieve terms that are difficult to find using classic NLP methods. The tf-idf score of this set of terms is set to `config[ 'Annotation' ].get( 'TF_IDF_REGEX' )`. 
    
5. Whitelist + blacklist:

    In `media/whitelist.txt` and `media/blacklist.txt`, users can provide a blacklist and a whitelist of terms. Blacklisted terms will be removed from the term list found in step 3, and whitelisted terms will be added to the list. The tf-idf score of whitelisted terms is set to `config[ 'Annotation' ].get( 'TF_IDF_WHITELIST' )`. The tf-idf score of terms found in step 4 that are also in the whitelist remains `config[ 'Annotation' ].get( 'TF_IDF_REGEX' )`.
    
5. Annotation of terms:

    Using the [Aho-Corasick algorithm](https://pypi.org/project/pyahocorasick/), the offsets of the terms found via step 2 to 5 are calculated. These offsets are then used to add the `config[ 'Annotation' ].get( 'TOKEN_TYPE' )` annotation to the cas. This feature type contains both the offsets, the lowercased version of the term and the tf-idf score ( <em> .begin, .end, .term, .tfidfValue </em> ).
    
6. Annotation of lemmata:

    Lemmata of terms are obtained using the Spacy model `config[ 'Annotation' ].get( 'SPACY_MODEL' )`. Lemmata annotations are added at the obtained offset of the original term using the `config[ 'Annotation' ].get( 'LEMMA_TYPE' )` feature type  ( <em> .begin, .end, .value </em> ). 
    

## Testing:

Unit and integration tests could be run with the unittest module or with pytest:

pytest $test_name.py