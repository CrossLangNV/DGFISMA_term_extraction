import os
from typing import List, Dict

import fasttext
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

from similar_terms.methods import SimilarWordsRetriever

ROOT = os.path.join(os.path.dirname(__file__), '../..')
FILENAME_FASTTEXT_MODEL = os.environ.get('FASTTEXT_PATH', os.path.join(ROOT, 'media/dgf-model.tok.bin'))
FASTTEXT_MODEL = fasttext.load_model(FILENAME_FASTTEXT_MODEL)

THRESHOLD = .8


class SingleVoc(BaseModel):
    voc: List[str]


class DoubleVoc(BaseModel):
    voc1: List[str]
    voc2: List[str]


@app.get("/")
async def read_main():
    return {"msg": "Similar term retrieval"}


@app.post("/similar_terms/align/")
async def align_vocs(vocs: DoubleVoc) -> Dict[str, str]:
    similar_words_retriever = SimilarWordsRetriever(fasttext_model=FASTTEXT_MODEL)

    try:
        similar_words_retriever.set_vocabulary(vocs.voc2)

        d_matches = {}
        for label in vocs.voc1:
            sim_words = similar_words_retriever.get_similar_thresh(label, thresh=THRESHOLD)

            sim_terms = sim_words['original terms']
            if len(sim_terms):
                d_matches.setdefault(label, []).extend(sim_terms)

        return JSONResponse(d_matches)

    except:
        raise HTTPException(status_code=406, detail='Incorrect json data.')


@app.post("/similar_terms/self/")
async def align_voc_self(vocs: SingleVoc) -> Dict[str, str]:
    result = align_vocs(DoubleVoc(voc1=vocs.voc, voc2=vocs.voc))

    return await result
