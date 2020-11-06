#FROM ubuntu:18.04
#gpu
FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

MAINTAINER arne <arnedefauw@gmail.com>

#ARG MODEL_DIR

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
RUN bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y python=3.7.3 && \
#conda install --name base scikit-learn=0.20.0 && \
conda install pandas=1.0.1 && \
conda install pytorch==1.7.0 cudatoolkit=10.1 -c pytorch && \
conda install -c conda-forge spacy==2.3.2 &&\
conda install -c conda-forge spacy-model-en_core_web_lg

#Install Cython
RUN apt-get update
RUN apt-get -y install --reinstall build-essential
RUN apt-get -y install gcc
RUN pip install Cython

RUN pip install \
Django==3.0.5 \
django-braces==1.14.0 \
django-cleanup==4.0.0 \
django-cors-headers==3.2.1 \
django-crispy-forms==1.9.0 \
django-oauth-toolkit==1.3.2 \
django-rest-framework-social-oauth2==1.1.0 \
djangorestframework==3.11.0 \
contractions==0.0.25 \
nltk==3.5 \
cloudpickle==1.3.0 \
torchtext==0.5.0 \
scikit-learn==0.23.2 \
transformers==3.4.0 \
scipy==1.4.1 \
numpy==1.18.5 \
tensorflow==2.3.1 \
keras==2.4.3 \
bs4==0.0.1 \
beautifulsoup4==4.5.3 \
fasttext==0.9.2 \
dkpro-cassis==0.3.0 \
pytest==6.0.1 \
plac==1.2.0 \
seqeval==1.2.2 \
pexpect \
ipython \
jupyter \
jupyterlab

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

#WORKDIR /work
#copy code:
#COPY app.py /work
#COPY code_bert_classifier/src/*.py /work/code_bert_classifier/src/
#copy model:
#COPY $MODEL_DIR/*.pth /work/models/model.pth
#COPY $MODEL_DIR/config.json /work/models/
#COPY $MODEL_DIR/special_tokens_map.json /work/models/
#COPY $MODEL_DIR/tokenizer_config.json /work/models/
#COPY $MODEL_DIR/vocab.txt /work/models/

#CMD python /work/app.py

EXPOSE 8888

VOLUME /notebook
WORKDIR /notebook

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip='0.0.0.0'"]
