"""Ngram.py

This file defines the NgramsFinder class.
NgramsFinder object are fed text and can derive simple and complex possible terminology.

Input:
------
Raw text is given to a NgramsFinder object via the feedText method.

Output:
-------
Simple terminology is derived through the getSortedNgrams method
Complex terminology is derived through the getSortedSuperNgrams method

Modifiable parameters:
----------------------
self.superGramMaxSize
    Modifiable in __init__
    Defines the maximal number of non-stopwords words contained in a supergram
    
self.stopwords        
    Modifiable in __init__
    Stopwords are not considered words but as a glue between words or sequence of words
    Thus they are an essential part of supergram extraction
    
self.nonValidWords
    Modifiable in __init__
    Filters out undesirable elements
    Any n-gram containing a nonValidWords will never be in the final output
    
self.freqThreshold    
    Modifiable in feedText
    Defines a minimum frequency threshold for n-gram extraction
    
Filtering
    Modifiable in cleanText
    Remove common and irrelavant contructs
    Replace symbols to simplify the analysis
    
    Modifiable in hasNoSymbols
    Symbols to detect
    
    Modifiable in isStopword
    rules for what are stopwords
    
    Modifiable in isValidTerm
    rules for what are valid terms
    
Dependency:
-----------
-NLTK is needed for word_tokenize and stopwords.
    This means that one should run the python command NLTK.download()
    And download the word_tokenize model and the stopwords corpus.
-Spacy is used for POS tagging
    To run the code one thus needs to run this command :
    "python -m spacy download en_core_web_sm"
-The utils import refers to a local file utils.py

Author:
-------
Judicael POUMAY (judicael.poumay@gmail.com)
"""

#Dependencies
from nltk.tokenize  import  word_tokenize
from nltk.corpus    import stopwords
from statistics     import mean 
from . import utils
import contractions
import numpy as np
import spacy
import nltk
import math
import re

#Constants
POSTAGGER                   = spacy.load('en_core_web_lg')
NGRAM_FREQ                  = 0
NGRAM_POS                   = 1
NGRAM_SIZE                  = 2
NGRAM_TOKENS                = 3
NGRAM_CACHE_JACCARD         = 4
NGRAM_CACHE_NORMFREQUENCY   = 5
NGRAM_CACHE_NORMPMI         = 6
NGRAM_CACHE_PMI             = 7
NGRAM_CACHE_FREQUENCY       = 8
NGRAM_CACHE_DICE            = 9
NGRAM_CACHE_NORMDICE        = 10
    
    
##########################        
#ngramsFinder
##########################
class NgramsFinder:
    def __init__(self,nMax):
        """
        Args
        ----            
            nMax, 
                Indicates the maximal size of simple terms to look for
                    nMax=3 means that feedText will look for 2-grams and 3-grams
                    nMax=4 means that feedText will look for 2-grams, 3-grams, and 4-grams
                Note that longer terms will be derived via supergram extraction
                    Thus we recommend to use nMax<=4.
                Also note that n refers to the number of non-stopwords words.
                    Thus the size of "amount of assets" is two
        """
        
        #Main objects members
        self.nMAx               = nMax+1
        self.stopwords          = set(stopwords.words('english')) 
        self.ngrams             = [{} for j in range(self.nMAx)]  
        self.superNgram         = {}
        self.nonValidWords      = set()
        self.freqThreshold      = 0
        self.superGramMaxSize   = 10
        
        #Caches (used for memoisation)
        self.supergrams         = []
        self.cleanedText        = ""        
        self.tokenedData        = []       
        self.maxFreq            = [-1 for j in range(self.nMAx)]
        self.maxPMI             = [-1 for j in range(self.nMAx)]
        self.maxDICE            = [-1 for j in range(self.nMAx)]
        self.docSize            = 0
        self.tokenFrequency     = {}
        
        #stopwords
        self.stopwords.remove("own")
        self.stopwords.add("less")
        self.stopwords.add("more")
        self.stopwords.add("regardless")
        self.stopwords.add("without")
        self.stopwords.add("whereas") #alina
        self.stopwords.add("therefore") #alina
        self.stopwords.add("except") #alina
        self.stopwords.add("hereto") #alina
        self.stopwords.add("where") #alina
        self.stopwords.add("which") #alina
        self.stopwords.add("whereof") #alina
        self.stopwords.add("since") #alina
        self.stopwords.add("hereby") #alina
        
        self.stopwords.add("due")
        self.stopwords.add("thereof")
        self.stopwords.add("please")
        self.stopwords.add("with")
        self.stopwords.add("also")  
        self.stopwords.add("would")  
        self.stopwords.add("whose")  
        self.stopwords.add("yet")  
        self.stopwords.add("is")  
        self.stopwords.add("'s")
        self.stopwords.add("(")
        self.stopwords.add(")")
        self.stopwords.add("[")
        self.stopwords.add("]")
        self.stopwords.add("{")
        self.stopwords.add("}")
        self.stopwords.add("be")
        self.stopwords.add("to")
        
        #noisy words
        self.nonValidWords.add("fields")
        self.nonValidWords.add("field")
        self.nonValidWords.add("sections")
        self.nonValidWords.add("chapter")
        self.nonValidWords.add("chapters")
        self.nonValidWords.add("section")
        self.nonValidWords.add("articles")
        self.nonValidWords.add("article")
        self.nonValidWords.add("table")
        self.nonValidWords.add("annex")
        self.nonValidWords.add("shall")
        self.nonValidWords.add("whether")
        self.nonValidWords.add("subparagraph") 
        self.nonValidWords.add("paragraph")  
        self.nonValidWords.add("where")  
        self.nonValidWords.add("referred")  
        self.nonValidWords.add("within")  
        self.nonValidWords.add("may")  
        self.nonValidWords.add("is")  
        self.nonValidWords.add("not")  
        self.nonValidWords.add("according")  
        self.nonValidWords.add("accordance")  
        self.nonValidWords.add("qualify")  
        
        #noisy symbols
        self.nonValidWords.add("january")
        self.nonValidWords.add("february")
        self.nonValidWords.add("march")
        self.nonValidWords.add("april")
        self.nonValidWords.add("may")
        self.nonValidWords.add("june")
        self.nonValidWords.add("july")
        self.nonValidWords.add("august")
        self.nonValidWords.add("september")
        self.nonValidWords.add("october")
        self.nonValidWords.add("november")
        self.nonValidWords.add("december")
        self.nonValidWords.add("one")
        self.nonValidWords.add("two")
        self.nonValidWords.add("three")
        self.nonValidWords.add("four")
        self.nonValidWords.add("five")
        self.nonValidWords.add("six")
        self.nonValidWords.add("seven")
        self.nonValidWords.add("eight")
        self.nonValidWords.add("nine")
        self.nonValidWords.add("ten")
        self.nonValidWords.add("''")
        self.nonValidWords.add("``")
        self.nonValidWords.add("`")
        self.nonValidWords.add("(-)")
        self.nonValidWords.add("a.")
        self.nonValidWords.add("b.")
        self.nonValidWords.add("c.")
        self.nonValidWords.add("d.")
        self.nonValidWords.add("e.")
        self.nonValidWords.add("f.")
        self.nonValidWords.add("g.")
        self.nonValidWords.add("h.")
        self.nonValidWords.add("i.")
        self.nonValidWords.add("j.")
        self.nonValidWords.add("k.")
        self.nonValidWords.add("l.")
        self.nonValidWords.add("m.")
        self.nonValidWords.add("n.")
        self.nonValidWords.add("o.")
        self.nonValidWords.add("p.")
        self.nonValidWords.add("q.")
        self.nonValidWords.add("r.")
        self.nonValidWords.add("s.")
        self.nonValidWords.add("t.")
        self.nonValidWords.add("u.")
        self.nonValidWords.add("v.")
        self.nonValidWords.add("w.")
        self.nonValidWords.add("x.")
        self.nonValidWords.add("y.")
        self.nonValidWords.add("z.")
        self.nonValidWords.add("'")
        self.nonValidWords.add("\"")
        self.nonValidWords.add("ii")
        self.nonValidWords.add("iii")
        self.nonValidWords.add("iv")
        self.nonValidWords.add("vi")
        self.nonValidWords.add("vii")
        self.nonValidWords.add("viii")
        self.nonValidWords.add("oj")
        self.nonValidWords.add("ix")
        self.nonValidWords.add("xi")
        self.nonValidWords.add("third")
        self.nonValidWords.add("fourth")      
        self.nonValidWords.add("cr")
        self.nonValidWords.add("equ")      
        self.nonValidWords.add("irb")      
        self.nonValidWords.add("columns")      
        self.nonValidWords.add("column")      
        self.nonValidWords.add("rows")      
        self.nonValidWords.add("row")      
        self.nonValidWords.add("item")      
        self.nonValidWords.add("items")      
        self.nonValidWords.add("point")      
        self.nonValidWords.add("points")      
        

    #---------------------------   
    #Ngram discovery
    #---------------------------   
    def feedText(self,text):   
        """
        Task
        ----
            For all n in [2,nMax], derives n-grams from raw text
        Args
        ----
            text,
                Text from which to extracts terms
        """     
        #empty caches
        self.top        = [[] for j in range(self.nMAx)]
        self.maxFreq    = [-1 for j in range(self.nMAx)]
        self.maxPMI     = [-1 for j in range(self.nMAx)]
        self.maxDICE    = [-1 for j in range(self.nMAx)]
        for n in range(len(self.ngrams)):
            for ngram in self.ngrams[n]:
                self.ngrams[n][ngram][NGRAM_CACHE_JACCARD]          = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY]    = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI]          = -1
                self.ngrams[n][ngram][NGRAM_CACHE_PMI]              = -1
                self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY]        = -1
                self.ngrams[n][ngram][NGRAM_CACHE_DICE]             = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE]         = -1
                
        #init
        clean_data, tokens              = self.cleanText(text) 
        self.docSize        = len(tokens)
        self.freqThreshold  = (self.docSize//30000) + 2
        print("Doc size : ",self.docSize)
        print("Dynamic frequency treshold choosen => ", self.freqThreshold)
            
        #find ngrams for n 1->self.nMAX
        print("discovering n-grams")
        for i in range(1,self.nMAx):
            wordsVector     = [0 for j in range(i)]
            stopwordsMatrix = [[] for j in range(i)]            
            #for each word either add to stopwords list 
            #or shift wordsVector and stopwordsMatrix content 
            #and add new ngram to dict
            for pos,t in enumerate(tokens):
                if self.isStopword(t):
                    stopwordsMatrix[-1].append(t)
                else:
                    self.shift(t,wordsVector,stopwordsMatrix,i)
                    if wordsVector[0] != 0:
                    
                        #check for symbols
                        hasNoSymbols = self.hasNoSymbols(wordsVector)
                        if(hasNoSymbols):
                            for stopwordsVector in stopwordsMatrix:
                                if(not self.hasNoSymbols(wordsVector)):
                                    hasNoSymbols = False
                                    break
                        
                        #add entry if no symbols                    
                        if hasNoSymbols:
                            self.addEntry(wordsVector,stopwordsMatrix,i,pos)
        
        return clean_data 
        
    def fuse(self,wordsVector,stopwordsMatrix):
        """
        Task
        ----
            Derives a string from a sequence of words and stopwords
        Args
        ----
            wordsVector,
                List of words
                
            stopwordsMatrix,
                List of list of stopwords
        Output
        ------
            The resulting string
            Its size
            And its tokens
        """
        #init
        res     = ""
        size    = len(wordsVector)
        tokens  = []
        
        #fuse wordsVector and stopwordsMatrix properly
        for i,(w,sws) in enumerate(zip(wordsVector,stopwordsMatrix)):            
            #add word
            if(len(w) > 0):
                res += w +" "
                tokens.append(w)
            
            #add stopwords (ignores last column)
            if i < len(wordsVector)-1:
                size += len(sws)
                for sw in sws:
                    if(len(sw) > 0):
                        res += sw+" "
                        tokens.append(sw)
                    
        #remove last space and return
        return res[0:-1],size,tokens
        
    def getPOS(self,text):
        """
        Task
        ----
            A a string from a sequence of words and stopwords
        Args
        ----
            text,
                string to analyze
        Output
        ------
            The tokens of the strings
            And their associated POS
        """
        tokens  = []
        tags    = []
        doc     = POSTAGGER(text)
        for token in doc:
            tokens.append(token.text)
            tags.append(token.pos_)
        return tokens, tags
              
    def addEntry(self,wordsVector,stopwordsMatrix,i,pos):  
        """
        Task
        ----
            Adds a new n-gram in the dictonnary from a sequence of words and stopwords if valid
        Args
        ----
            wordsVector,
                List of words
                
            stopwordsMatrix,
                List of list of stopwords
                
            i, 
                Size of n-gram to add, that is i=n in n-gram
            
            pos,
                Position of n-gram to store
        """      
        #get fused ngram
        ngram, size, tokens = self.fuse(wordsVector,stopwordsMatrix)              
        
        ###
        #added grammar check Alina
        doc = POSTAGGER(ngram)
        labels = ['ADP', 'VERB', 'PRON', 'CCONJ', 'SCONJ' ]
        #if any(word.pos_ in labels for word in doc) == False  and any(word in self.stopwords for word in doc) == False:
        if any(word.pos_ in labels for word in doc) == False:
            chunks = list(doc.noun_chunks)
            if len(chunks) > 0:
                last_chunk = list(doc.noun_chunks)[-1]
                last_chunk = str(last_chunk)
                ngram = (ngram.split(last_chunk)[0] + last_chunk)
        ###
        
        #If exist, increment count and add new position
        if ngram in self.ngrams[i].keys():
            self.ngrams[i][ngram][NGRAM_FREQ] += 1
            self.ngrams[i][ngram][NGRAM_POS].append(pos-size+1)
        #Else if new and valid, set count, add position, set size, set tokens, and init caches
        else:
            if(self.isValidTerm(ngram,tokens)):            
                tok, tags = self.getPOS(ngram)
                if((tags[0] != "VERB" or not tok[0].endswith("ed")) #alina
                    and (tags[-1] != "VERB" or not tok[-1].endswith("ed"))): #alina                                     
                        self.ngrams[i][ngram] = [1,
                                                [pos-size+1],
                                                size,
                                                tokens,
                                                -1,-1,-1,-1,-1,-1,-1]
                
    def shift(self,word,wordsVector,stopwordsMatrix,i):  
        """
        Task
        ----
            Shifts the wordsVector and stopwordsMatrix
        Args
        ----
            wordsVector,
                List of words
                
            stopwordsMatrix,
                List of list of stopwords
                
            i, 
                Size of n-gram being dealt with, that is i=n in n-gram
        """            
        #shift all content
        for j in range(i):
            if(j > 0):
                stopwordsMatrix[j-1] = stopwordsMatrix[j]
                wordsVector[j-1]     = wordsVector[j]   
                
        #set last
        stopwordsMatrix[-1] = []
        wordsVector[-1]     = word
    
    #-------------    
    #Getter
    #-------------    
    def getTokenPos(self,ngram,dict):
        """
        Task
        ----
            Return the list of position for a given ngram
        Args
        ----
            ngram,
                ngram being considered
                
            dict,
                Dictionnary containing the ngram
            
        Output
        ------
            The list of position of the ngram being considered
        """            
        return dict[ngram][NGRAM_POS]
        
    
    #---------------------
    #Get sorted results
    #---------------------
    def getSortedSuperNgrams(self):
        """
        Task
        ----
            List supergram sorted and filtered by frequency 
        Output
        ------
            Returns a list of complex terms sorted by frequency
        """   
        if(self.supergrams):
            return self.supergrams
            
        #get supergrams
        for k in self.superNgram.keys():
            self.superNgram[k][1] = list(set(self.superNgram[k][1]))
        res = [(self.superNgram[k][0],k) for k in self.superNgram 
                    if self.superNgram[k][0] >= self.freqThreshold]
            
        #return sorted
        res = sorted(res, key=lambda tup: tup[0], reverse = True)
        
        self.supergrams = [x[1] for x in res]
        return self.supergrams
        
    def getSortedNgrams(self,n, func, filterSuperGram = True): 
        """
        Task
        ----
            List supergram sorted by a given measure filtered by frequency and by supergram subset test.
        Args
        ----
            n,
                size of n-grams being considered
                
            func,
                measure to use for ranking
                
            filterSuperGram,
                wether or not we should use supergram subset test
                If yes, n-grams that are part of supergrams are filtered out
                
        Output
        ------
            Returns a list of simple terms sorted by a given measure
        """   
        if(filterSuperGram):
            supergrams = self.getSortedSuperNgrams()
        else:
            supergrams = []
            
        #aplly frequency and part of term filter  
        filteredNgrams = []
        for ngram in self.ngrams[n].keys():
            if(self.getFreq(n,ngram) >= self.freqThreshold):
                filteredNgrams.append(ngram)
        
        
        #score ngrams
        res = []
        for ngram in filteredNgrams:  
            score = func(n,ngram)    
            res.append((score,ngram))
            
        #return sorted
        return sorted(res, key=lambda tup: tup[0], reverse = True)
    
        
    #------------------------
    #Text processing
    #------------------------
    def cleanText(self,data):  
        """
        Task
        ----
            Prepares the data for analysis by lowering letter, replacing constructs,add spaces, ...
            Finally the data is tokenized.
        Args
        ----
            data,
                raw data to clean
                
        Output
        ------
            Tokenized and cleaned raw data
        """ 
        #check cache
        if(self.tokenedData):
            return self.tokenedData
            
        #clean
        data = data.lower()
        data = contractions.fix(data)       

        
        data = data.replace('implementing technical standards with regard to supervisory reporting of institutions according to regulation','')
        data = data.replace('exposuresthe','exposures. the')
        data = data.replace('/',' / ')
        data = data.replace('.',' . ')
        data = data.replace('‘',' ‘ ')
        data = data.replace('’',' ’ ')
        data = data.replace(',',' , ')
        data = data.replace(':',' : ')
        data = data.replace(' - ',' |-| ')
        data = data.replace('as defined in',' | ')
        data = data.replace(' eur ',' | ')
        #data = data.replace('\r',' | ')
        #data = data.replace('\n',' | ')
        
        #cache and return
        self.cleanedText = data        
        self.tokenedData = self.tokenize(data) # Alina
        return self.cleanedText, self.tokenedData
        
    def tokenize(self,txt):
        """
        Task
        ----
            Tokenize a given piece of text.
            Uses NLTK tokenization and improves on it by fusing hyphen linked words and signle elements in parenthesis.
        Args
        ----
            data,
                Raw data to clean
                
        Output
        ------
            Tokenized text
        """ 
        #simple tokenization
        tokens = word_tokenize(txt)  
        
        #Fuse bracketed words with their surrouding bracket tokens
        toDel = []
        for i,t in enumerate(tokens):
            if(i > 0 and i < len(tokens)-1):
                if(tokens[i-1]  == "(" and tokens[i+1] == ")" 
                        or tokens[i-1] == "[" and tokens[i+1] == "]" 
                        or tokens[i-1] == "{" and tokens[i+1] == "}"):
                    tokens[i]   =  tokens[i-1]+tokens[i]+tokens[i+1]
                    toDel.append(i-1)
                    toDel.append(i+1)
                    
                if(tokens[i] == "-"):                    
                    tokens[i] = tokens[i-1]+tokens[i]+tokens[i+1]
                    toDel.append(i-1)
                    toDel.append(i+1)
                    
        #remove bracket tokens
        toDel.reverse()
        for d in toDel:
            del tokens[d]
            
        return tokens
        
    #----------------
    #Checks
    #----------------
    def isStopword(self,w):
        """
        Task
        ----
            Check if a word is a stopwords using the stopwords list, digit detection, ...
        Args
        ----
            w,
                word to check
                
        Output
        ------
            Boolean, true if w is a stopwords
        """ 
        if w in self.stopwords:
            return True
        if(any(c.isdigit() for c in w)): 
            return True
        if("(" in w and ")" in w and " " in w):
            return True
        return False
        
    def isValidTerm(self,ngram,tokens):
        
        """
        Task
        ----
            Check if a ngram is valid
        Args
        ----
            ngram,
                ngram to check
                
            tokens,
                tokenized ngram
                
        Output
        ------
            Boolean, true if w is valid
        """ 
        #check if number
        # if(any(c.isdigit() for c in ngram)): return False
        
        #check if too small token
        # for t in self.tokenize(ngram):
            # if(len(t) < 2):
                # return False
            
        #check if invalid token
        for w in self.nonValidWords:
            if(w in tokens):
                return False
        return True
        
    def hasNoSymbols(self,tokens):
        """
        Task
        ----
            Check if a list of tokens contains symbols
            
        Args
        ----
            tokens,
                tokenized text
                
        Output
        ------
            Boolean, true if tokens contains symbols
        """ 
        symbols = ["″","‘","–","’",".",",",";",":","?","!","(",")",
                    "[","]","{","}","-","—","_","$","£",
                    "€","|","&","#","%","+","*","/","\\","<",">"]
        for s in symbols:
            for l in tokens:
                if(s in l and len(l) < 4):
                    return False
        return True
            
    #----------------
    #Ngram Measures
    #----------------     
    """
    Task
    ----
        Each function computes a measure
        
    Args
    ----
        n,
            size of ngram being measured
            
        ngram,
            ngram being measured
            
    Output
    ------
        Returns the value of the measure applied to the ngram being considered
        
    Measures defined     
    ----------------
        Frequency
        Normalized frequency
        PMI
        Normalized PMI
        C-value
        One, returns 1
        ISW, measure designed for noise measurement
        CPS², Measure mix designed for simple terminology extraction
    """ 
    def getInvFactStopwords(self,n,ngram):
        tokens = self.ngrams[n][ngram][NGRAM_TOKENS]
        
        #count nb stopwords
        res = 0        
        for t in tokens:
            if self.isStopword(t):
                res += 1
               
        #normalize and return
        return (1-res/len(tokens))  
        
    def getFreq(self,n,ngram): 
        if(self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY]
            
        res = self.cleanedText.count(ngram+" ")
        res += self.cleanedText.count(ngram+".")
        
        self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY] = res
        return self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY]
        
        
    def getNormalizedFreq(self,n,ngram): 
        if(self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY]
    
        if(self.maxFreq[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getFreq(n,k) > max):
                    max = self.getFreq(n,k)
            self.maxFreq[n] = max
        #self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] = self.getFreq(n,ngram)/self.maxFreq[n]
        self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] = self.getFreq(n,ngram)
        return self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] 
        
    def getCPS2(self,n,ngram):
        isw = self.getInvFactStopwords(n,ngram)             
        c = self.getCvalue(n,ngram)             
        pmi = self.getNormalizedPMI(n,ngram)            
        #return c*pmi*isw*isw
        return c
        
    def getNormalizedPMI(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI]
                
        if(self.maxPMI[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getPMI(n,k) > max):
                    max = self.getPMI(n,k)
            self.maxPMI[n] = max
        
        self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI] = self.getPMI(n,ngram)/self.maxPMI[n]
        return self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI]
    
    def getPMI(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_PMI] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
            
        fAll    = self.getNormalizedFreq(n,ngram)
        if(fAll == 0):
            self.ngrams[n][ngram][NGRAM_CACHE_PMI] = 0
            return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
        tokens  = [t for t in self.ngrams[n][ngram][NGRAM_TOKENS] 
                        if not self.isStopword(t)]
        N       = len(self.tokenedData)    
        fs      = [self.getTokenFrequency(n,t) for t in tokens]
        denum   = 1
        for f in fs:
            denum*=f/N
            
        self.ngrams[n][ngram][NGRAM_CACHE_PMI] = math.log(fAll/(denum*N))
        return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
        
    def getOne(self,n,ngram):                  
        return 1
    
    def getCvalue(self,n,ngram):                  
        fAll        = self.getNormalizedFreq(n,ngram)     
        superList   =  []
        for i in range(n+1,self.nMAx):
            res = self.getSortedNgrams(i,self.getOne,False)
            for elem in res:
                if(ngram in elem[1]):
                    superList.append(elem[0])
        if(not superList):
            superList.append(0)
        return fAll-mean(superList)
    
    def getTokenFrequency(self,n,token):
        """ Auxillary function for PMI """
        if(token in self.tokenFrequency):
            return self.tokenFrequency[token]
            
        if(self.maxFreq[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getFreq(n,k) > max):
                    max = self.getFreq(n,k)
            self.maxFreq[n] = max
            
        res = self.cleanedText.count(token)/self.maxFreq[n]
                
        self.tokenFrequency[token] = res
        return self.tokenFrequency[token]