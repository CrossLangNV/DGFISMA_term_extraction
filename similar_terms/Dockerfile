##FROM python:3.7
#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
#
#LABEL maintainer="laurensmeeus93@gmail.com"
#
##RUN pip install fastapi uvicorn
#
#COPY ./requirements.txt /tmp/
#RUN pip install --no-cache-dir -r /tmp/requirements.txt
#
##EXPOSE 80
#
#COPY ./ /similar_terms
#COPY ./app /app/app



FROM python:3.7

LABEL maintainer="laurensmeeus93@gmail.com"

COPY similar_terms/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 80

COPY similar_terms /similar_terms

COPY media/dgf-model.tok.bin /media

#COPY ./app /app/app

CMD ["uvicorn", "similar_terms.app.main:app", "--host", "0.0.0.0", "--port", "80"]

