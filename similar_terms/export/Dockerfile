FROM python:3.7

LABEL maintainer="laurensmeeus93@gmail.com"

COPY similar_terms/requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# To get our RDF packages
COPY DGFISMA_RDF/ /tmp/DGFISMA_RDF/
WORKDIR /tmp/DGFISMA_RDF
RUN python /tmp/DGFISMA_RDF/setup.py install
WORKDIR /

EXPOSE 80

COPY similar_terms /similar_terms
COPY user_scripts/ /user_scripts

COPY media/eurovoc.py /media/eurovoc.py

CMD ["uvicorn", "similar_terms.export.app.main:app", "--host", "0.0.0.0", "--port", "80"]

