import os
from typing import List, Dict

import fasttext
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse  # ,FileResponse
from pydantic import BaseModel

from media.data import get_filename_fasttext_model
from similar_terms.methods import SimilarWordsRetriever

app = FastAPI()

FILENAME_FASTTEXT_MODEL = os.environ.get('FASTTEXT_PATH', get_filename_fasttext_model())
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
