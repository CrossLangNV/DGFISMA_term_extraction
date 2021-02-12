<h1> Bert Bio tagging </h1>

BIO tagging, in the context of the DGFISMA project is used for determining which concept is defined by a certain definition. E.g., given a definition, detected by our definition detector: <br />

```
For the purpose of this paragraph annual contribution means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```


We want to tag the term `annual contribution` with the tags B and I. All other words in the sentence should be tagged as O, for instance the term `administrative expenditures of the Board` which is not defined by this definition, should be tagged as O O O O O.

The model used for BIO tagging is the BertForTokenClassification model ( https://github.com/huggingface/transformers ), which consists of a Bert Model with a Token classification layer on top. 

<h2> Preparation of training data </h2>

Given a definition, containing a term, e.g. 

``` 
For the purpose of this paragraph annual contribution means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```

The annotator should annotate the term that is defined by this definition with the special symbols ★ and ☆, or two other symbols that are in the Bert vocabulary, and are not expected to occur in the training data. The annotator should also make sure to leave a space between the special symbols ★ and ☆ and other words. The above sentence thus becomes:

``` 
For the purpose of this paragraph ★ annual contribution ☆ means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```

If some quotation is present around the concept defined by the definition, e.g.: 

```
For the purpose of this paragraph 'annual contribution' means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```

, then we advice to annotate as follows:

```
For the purpose of this paragraph ★ 'annual contribution' ☆ means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```


The provided code will strip the quotations from the annotated term, to prevent the model to memorize these special symbols. The list of special symbols that are stripped is: [ " , ‘ , " , ` , ' , ’ , • , “ , ‧ ], and all symbols that are not present in the Bert vocabulary.

The annotated data should be saved in a simple text file separated by newlines. 

We provide code to generate training data from these annotated sentences. From python interpreter run:

```
from user scripts import generate_training_data
generate_training_data.main( "training_set_annotated.txt",\
                            "training_set_bio_tag.csv",\
                            seq_length=75 )
```
With <em> training_set_annotated.txt </em> the annotated training data, and <em> training_set_bio_tag.csv </em> a resulting csv file with BPE-tokenized (BertTokenizer) tokens and BIO tags, that can be used by the provided training script (next section).

The sequence length is the cut off value for the number of tokens in the (Bert)tokenized sentence. Note that Bert supports sequences of up to 512 tokens.

The symbols used for annotation are set by default to ★ and ☆.

<h3> Example </h3>

Given a training set consisting of the sentences:

<em> ★ ' Asset ' ☆ means something. <br/>
The ★ Profit ☆ is defined as something else.</em>

Then the csv file produced by the user script `generate_training_data`, will be:


Sentence #  | Word | POS | Tag 
--- | --- | --- |--- |
Sentence 1 |  asset | -- | B | 
 / |  means | -- | O |
 / |  something | -- | O |
 / |  . | -- | O |
 Sentence 2 |  the | -- | O | 
 /|  profit | -- | B |
 / |  estimate | -- | I |
 /|  is | -- | O |
 /|  defined | -- | O |
 /|  as | -- | O |
 /|  something | -- | O |
/|  else | -- | O |
/|  . | -- | O |

Note the lowercasing and how quotation is stripped from the annotated term. 

<h2> Preparation of training data using annotations in UIMA CAS object </h2>

We provide a user script to convert annotations in a CAS object, provided by the users of the Glossary app to the format that can be used by the `generate_training_data.py` user script.

The annotations added by the users of the Glossary app are: `config[ 'Annotation_user' ].get( 'TOKEN_TYPE_USER' )`, `config[ 'Annotation' ].get( 'DEFINED_TYPE_USER' )` and `config[ 'Annotation' ].get( 'DEFINITION_TYPE_USER' )`. 

Using the user script `generate_training_data_CAS.py`, and given a directory containing CAS objects (<em>.xmi</em> format), we can run the following code from the Python Interpreter:

```
from user_scripts import generate_training_data_from_cas
generate_training_data_from_cas.main( "{Path}/dir_cas" ,\
                                      "{Path}/typesystem_user.xml" ,\
                                      "{Path}/TermExtraction.config",\
                                      "{Path}/whitelist_terms.txt",\
                                      "{Path}/training_set_def.txt",\
                                      "{Path}/training_set_def_test.processed.txt",
                                    users=[ 'user1', 'user2'  ]
                                    )
```

with, relative to directory path {Path}, <em>dir_cas</em> the directory containing the CAS objects, <em>typesystem_user.xml</em> the used typesystem, <em>TermExtraction.config</em> the configuration file. 

The script will generate three files: <em>whitelist_terms.txt</em>, <em>training_set_def.txt</em> and <em>training_set_def.processed.txt</em>. 

- <em>whitelist_terms.txt</em>: a list of whitelisted terms annotated by the users of the Glossary app with `config[ 'Annotation_user' ].get( 'TOKEN_TYPE_USER' )`). 
- <em>training_set_def.txt</em>: a list of definitions that can be used to train [DistilBertSequenceClassifier](https://github.com/CrossLangNV/DGFISMA_definition_extraction) for detection of definitions. The list of extracted definitions is obtained via `config[ 'Annotation_user' ].get( 'VALUE_BETWEEN_TAG_TYPE' )`) with tagname=<em>'p'</em> containing a `config[ 'Annotation' ].get( 'DEFINED_TYPE_USER' )` annotation.
- <em>training_set_def.processed.txt</em>: a list of annotated definitions obtained using the `config[ 'Annotation' ].get( 'DEFINED_TYPE_USER' )` annotation.

<h2> Training </h2>

We provide code to train the classification layers of a BertForTokenClassification model:

```
from user_scripts import train
train.main( "training_set_bio_tag.csv",\
             "bio_tagger",\
             seq_length=75,\
             epochs=20)
```

With <em> training_set_bio_tag.csv </em> the file created by the user script `generate_training_data`.

This will train the model, create the directory <em> bio_tagger </em>, and save the model there (<em>config.json</em>, <em>pytorch_model.bin</em>, <em>vocab.txt</em> and the pickle <em>tags_vals</em>, containing a python list, that allows us to make the conversion between BertForTokenClassification prediction id's back to BIO tag labels).
The model will train on GPU if available, and if not, on CPU. 

<h2> Inference </h2>

The trained model can be used to predict the BIO tags of tokens in a set of sentences:

```
from user_scripts import inference
inference.main(  "test_sentences.txt",\
             "bio_tagger",\
             "results_test_sentences.csv"
              )
```
              
With <em> bio_tagger </em> the directory where the trained BertForTokenClassification model is stored, and <em> test_sentences.txt </em> a text file containing sentences. Note that these sentences do not need to be tokenized, they will be tokenized during inference. Results will be written to the csv_file <em> results_test_sentences.csv </em> (.csv file has same format as described in the example above). 

<h2> Evaluation </h2>

Results produced by the inference script can be evaluated, given a csv file in the same format containing Gold Standard BIO tags (e.g. <em> results-gold_standard.csv </em> ) for each token:

```
from user_scripts import evaluate
evaluate.main( "results_test_sentences.csv", \
                    "results-gold_standard.csv"
)
```

This will return a classification report with precision and recall scores using seqeval (https://pypi.org/project/seqeval), a python framework for sequence labeling evaluation.

The easiest way to obtain the Gold Standard BIO tags, is to run the user script `inference` and correct the resulting csv file.

<h2> Results </h2>

When trained on 296 annotated sentences (see release link), containing 10277 tokens, we obtain the following results on a held out test set (22 sentences, 857 tokens):

Model | precision | recall | f1-score | support |
--- | --- | --- |--- |--- |
BertForTokenClassification |  0.86 | 0.76 | 0.81 | 33 | 