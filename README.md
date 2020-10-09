# term-extraction

## Instructions

use "dbuild.sh" to build the docker image

use "dcli.sh" to start a docker container

Given a document [json](https://github.com/alina-crosslang/term-extraction/blob/master/example.json), the program will return a json containing annotated 'cas_content' and 'content_type.'
The "cas_content" is a UIMA CAS object, encoded in base64. The "content_type" can be "html" or "pdf". The "extract_supergrams" parameter could be either "true" or "false."


The Term Extraction pipeline consists of the following steps:

1. After decoding from base64, read in the CAS object using [the dkpro-cassis library](https://github.com/dkpro/dkpro-cassis)
2. Retrieve text segments:

   cas.get_view("html2textView").select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType") will return a generator containing all tags detected by UIMA. 
   We retrieve the text segments by limiting ourself to the 'p' tags.
   
3. Extract terms from text segments :
   
   SpaCy will process a raw text and return a Doc object with various annotations. 
   Doc.noun_chunks is a generator (list) containing all 'flat' noun phrases, i.e. phrases with a noun as their head.
   POS tags, as well as a list of invalid words, are used to filter the terms.
   Dependency values of nouns in the Doc are used for the extraction of complex noun phrases, i.e. supergrams. 
   With the head noun as root, its attributes left_edge and right_edge will provide the first and the last tokens of the subtrees.
   Supergrams are therefore Doc spans from the root's left_edge's index up to and including its right_edge.

4. Calculate tf-idf for the extracted terms

   We use sklearn's Tfidftransformer to calculate the tf-idf score per term per document. 
   The document in this context is a list of text segments.
   CountVectorizer() will count the number of occurrences of terms for all ngrams.
   TfidfTransformer will calculate the IDF on the CountVectorizer's term counts.
   We want to generate a matrix of term counts in case of previously unseen documents.
   Finally, the tf-idf score can be calculated.
   
5. Annotate the original CAS with terms, tf-idf scores and lemmata
6. Encode the annotated CAS to base64
7. Return a json with the annotated CAS and the original field 'content_type'

## Testing:

Unit and integration tests could be run with the unittest module or with pytest:

pytest $test_name.py
