"""
The FastText model for similar term retrieval is too big to be included in GitHub.
Instead it will be downloaded if it doesn't exist yet.
"""

import os

import requests

FILENAME_FASTTEXT_MODEL = os.path.join(os.path.dirname(__file__), 'dgf-model.tok.bin')
# URL_FASTTEXT_MODEL = 'https://drive.google.com/file/d/1Pjn8jAgd4gfFGQgoDzhgdoIwhvm0HNE_/view?usp=sharing'
FILE_ID = '1Pjn8jAgd4gfFGQgoDzhgdoIwhvm0HNE_'


def get_filename_fasttext_model():
    if not os.path.exists(FILENAME_FASTTEXT_MODEL):
        download_file_from_google_drive(FILE_ID, FILENAME_FASTTEXT_MODEL)

    return FILENAME_FASTTEXT_MODEL


def download_file_from_google_drive(id, destination):
    """

    Args:
        id: file_id, TAKE ID FROM SHAREABLE LINK
        destination: DESTINATION FILE ON YOUR DISK

    Returns:

    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    get_filename_fasttext_model()
