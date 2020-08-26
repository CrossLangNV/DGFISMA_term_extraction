from bs4 import BeautifulSoup, Comment, Doctype
import re
from typing import List, Tuple, Set

from cassis import Cas


def clean_html(  html_file  ):
    
    '''
    Function will find all text in the html, convert to a plain text (String), and will split up the String in sections using the section_regexp.
    
    :param html_file: String containing a html file in plain text.
    :return: List of Strings (articles). 
    '''

    section_regexp= r'^(Part|Title|Chapter|Section|Sub-Section|Article|ANNEX)\ *([0-9.]|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|I|II|III|IV|V|VI|VII|VIII|IX|X|X(I|II|III|IV|V|VI|VII|VIII|IX|X))* *$'

    
    page_content=BeautifulSoup( html_file, "html.parser")

    all_text=[]

    # remove the header:
    [x.extract() for x in page_content.findAll('head')]

    # remove the items of Doctype type:
    for item in page_content:
        if isinstance(item, Doctype):
            item.extract()

    # remove the comments
    com = page_content.findAll(text=lambda text:isinstance(text, Comment))
    [comment.extract() for comment in com]

    for node in page_content.findAll('p'):
        text = ''.join(node.findAll(text=True)) 
        text = text.strip() 
        text= text.replace( "\n", "" )
               
        if text:
            if bool (re.match( section_regexp, text , re.IGNORECASE )):
                all_text.append( 100*"■" )

            all_text.append( text )

    all_text="\n".join(all_text)

    articles=all_text.split( 100*"■" )
    return articles


def get_text_html(cas: Cas, SofaID: str, tagnames: Set[str] = set('p')) -> List[str]:
    '''
    Given a cas, and a view (SofaID), this function selects all ValueBetweenTagType elements ( with tag.tagName in the set tagnames ), extracts the covered text, and returns the list of extracted sentences.
    '''

    sentences = []
    for tag in cas.get_view(SofaID).select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        if tag.tagName in set(tagnames):
            sentence = tag.get_covered_text().strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def get_text_html(cas: Cas, SofaID: str, tagnames: Set[str] = set('p')) -> List[str]:
    '''
    Given a cas, and a view (SofaID), this function selects all ValueBetweenTagType elements ( with tag.tagName in the set tagnames ), extracts the covered text, and returns the list of extracted sentences.
    '''

    sentences = []
    for tag in cas.get_view(SofaID).select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        if tag.tagName in set(tagnames):
            sentence = tag.get_covered_text().strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def get_text_pdf(cas: Cas, SofaID: str) -> List[str]:
    '''
    Given a cas, and a view, this function should return the text we want. TO DO
    '''

    pass
