import os
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse

from user_scripts.similar_terms import export_glossary_link_eurovoc_rdf

app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Export Glossary and mapping with EuroVoc."}


@app.post("/export_sim_terms_eurovoc/")
def export_sim_terms_eurovoc(file: UploadFile = File(...)):
    """

    Args:
        file: A CSV with per row Term, Definition (, and Lemma) of a concept.

    Returns:

    """

    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filename_csv = os.path.join(tmp_dir, 'tmp.csv')
    filename_rdf = os.path.join(tmp_dir, 'export_sim_terms_eurovoc.turtle')



    with open(filename_csv, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    export_glossary_link_eurovoc_rdf.main(filename_terms=filename_csv,
                                          filename_rdf=filename_rdf)

    return FileResponse(filename_rdf,
                        media_type='rdf/turtle',
                        filename=os.path.basename(filename_rdf)
                        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
