# term-extraction


use "dbuild.sh" to build the docker image

use "dcli.sh" to start a docker container

Given a document (json), e.g.: https://github.com/alina-crosslang/term-extraction/blob/master/example.json, the program will return a json containing annotated 'cas_content' and 'content_type.'

The Term Extraction pipeline consists of the following steps:


1. Decode the original CAS content from base64
2. Convert HTML to text 
3. Extract terms from text segments (<p>) :
   
   SpaCy will process a raw text and return a Doc object with various annotations. 
   Doc.noun_chunks is a generator (list) containing all 'flat' noun phrases, i.e. phrases with a noun as their head.
   POS tags, as well as a list of invalid words, are used to filter the terms.
   Dependency values of nouns in the Doc are used for the extraction of complex noun phrases, i.e. supergrams. 
   With the head noun as root, its attributes left_edge and right_edge will provide the first and the last tokens of the subtrees.
   Supergrams are therefore Doc spans from the root's left_edge's index up to and including its right_edge.

4. Calculate tf-idf for the extracted terms

   We use sklearn's Tfidftransformer to calculate the tf-idf score per term per document. 
   The document in this context is a list of segments (<p>).
   CountVectorizer() will count the number of occurrences of terms for a given word range.
   TfidfTransformer will calculate the IDF on the CountVectorizer's term counts.
   We want to generate a matrix of term counts in case of previously unseen documents.
   Finally, the tf-idf score can be calculated.
   
5. Annotate the original CAS with terms, tf-idf scores and lemmata
6. Encode the annotated CAS to base64
7. Return a json with the annotated CAS and the original field 'content_type'