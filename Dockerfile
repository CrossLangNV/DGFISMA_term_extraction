FROM ubuntu:18.04

# Install some basic utilities
#RUN apt-get update && apt-get install -y apt-transport-https 
RUN apt-get -qq -y update
RUN apt-get -qq -y upgrade
RUN apt-get -qq -y install -y apt-transport-https \
    ca-certificates \
    curl \
    sudo \
	&& rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
RUN bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN apt-get update && apt-get dist-upgrade
#RUN apt-get -y install --reinstall build-essential &&
RUN apt-get -y install gcc
RUN apt-get -y install vim

RUN conda install -y python=3.7
RUN conda install -c conda-forge pyemd
RUN conda install -c conda-forge spacy
RUN conda install -c conda-forge spacy-model-en_core_web_lg

RUN pip install lxml==4.5.0 \
                Django==3.0.5 \
                django-braces==1.14.0 \
                django-cleanup==4.0.0 \
                django-cors-headers==3.2.1 \
                django-crispy-forms==1.9.0 \
                django-oauth-toolkit==1.3.2 \
                django-rest-framework-social-oauth2==1.1.0 \
                djangorestframework==3.11.0 \
                scikit-learn==0.22.2 \
                pandas==1.0.1 \
                ipython \
                nltk \
                dkpro-cassis==0.3.0 \
                pybase64 \
                jsonlines \
                contractions \
                bs4==0.0.1

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

#RUN python -m spacy download en_core_web_sm
#CMD python /django/manage.py makemigrations && python /django/manage.py migrate && python /django/manage.py runserver 0.0.0.0:5000
