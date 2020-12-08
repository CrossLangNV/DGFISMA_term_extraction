import pytest
from dgconcepts.pipeline.inference import clean_dictionary

import configparser

import os

def test_clean_dictionary():
        
    terms_dict=dict()
    terms_dict[ "11  " ]=0
    terms_dict[ "> <  " ]=0
    terms_dict[ "—" ]=0
    terms_dict[ "&" ]=0
    terms_dict[ "level a assets" ]=0
    terms_dict[ "level 1 assets" ]=0
    terms_dict["?test"]=0
    terms_dict[ "%^[ 4445 44.f" ]=0
    terms_dict[ "%," ]=0
    terms_dict[ "…" ]=0
    terms_dict[ "+" ]=0
    terms_dict[ "÷" ]=0

    clean_dictionary( terms_dict )
    assert terms_dict == {'level a assets': 0, 'level 1 assets': 0, '?test': 0, '%^[ 4445 44.f': 0}
    