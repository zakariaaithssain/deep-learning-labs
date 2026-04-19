import pandas as pd

import re
import string
import spacy

from bs4 import BeautifulSoup

DATA_PATH = "data/imdb_reviews.csv"



def preprocess(text: str):

    #remove html tags
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ', strip=True)
    text = text.lower()

    #urls
    text = re.sub(r'http\S+|www\S+', '', text)
    #digits
    text = re.sub(r'\d+', '', text) 
    #extra whitespaces 
    text = re.sub(r'\s+', ' ', text).strip()

    #punctuation
    table = str.maketrans('', '', z= string.punctuation)
    text = text.translate(table)

    #tokenization and removing stop words 
    nlp =spacy.load('en_core_web_sm')
    doc = nlp(text)
    clean_tokens = [token.text for token in doc if not token.is_stop]

    return clean_tokens