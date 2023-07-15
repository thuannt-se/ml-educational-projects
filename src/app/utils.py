import streamlit as st
import pandas as pd
import os
import numpy as np
import itertools
import seaborn as sns
import joblib
from joblib import dump
import pickle
#language processing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from sklearn.pipeline import Pipeline

#Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

#data cleaning
import re
import emoji
import string
from underthesea import word_tokenize

PATH_TO_MODEL = "../../models"
PATH_TO_PIPELINES = "../../pipelines"
STOP_WORDS_PATH = "../../data_set/ecommerce_review/vietnamese-stopwords.txt"
stop_words = []
with open(STOP_WORDS_PATH, 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # Skip empty lines
            stop_words.append(line)  # Add the line's element to the list

'''
 For removing breakline
'''
def re_breakline(texts):
    return [re.sub('[\n\r]', ' ',t) for t in texts ]

'''
 For removing punctuation, and space.
'''
def re_punctuation(texts):
    return  [" ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", str(t).lower()).split()) for t in texts]


'''
 For removing number.
'''
def re_numbers(texts):
    """
    Args:
    ----------
    texts: text content to be prepared [type: string]
    """
    # Applying regex
    return [re.sub('[0-9]+', '', str(t)) for t in texts]

def re_hiperlinks(texts):
    """
    Args:
    ----------
    texts: text content to be prepared [type: string]
    """
    # Applying regex
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' link ', t) for t in texts]

def re_emoiji(texts):
    """
    Args:
    ----------
    text: list object with text content to be prepared [type: string]
    """
    # Applying regex
    return [emoji.replace_emoji(t) for t in texts]


def re_whitespaces(texts):
    """
    Args:
    ----------
    text:  text content to be prepared [type: string]
    """

    # Applying regex
    white_spaces = [re.sub('\s+', ' ', t) for t in texts]
    white_spaces_end = [re.sub('[ \t]+$', '', w) for w in white_spaces]
    return white_spaces_end



def data_tokenizer(texts, stop_words = stop_words):
    return [' '.join([c for c in word_tokenize(t.lower()) if c not in stop_words]) for t in texts]

class ApplyRegex(BaseEstimator, TransformerMixin):

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying all regex functions in the regex_transformers dictionary
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)

        return X

class TextFeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer, train=True):
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.train:
            return self.vectorizer.fit_transform(X).toarray()
        else:
            return self.vectorizer.transform(X).toarray()



def predict(data):
    text_transformers = {
        'numbers': re_numbers,
        'emoiji': re_emoiji,
        'breakline': re_breakline,
        'punctuation': re_punctuation,
        'hiperlinks': re_hiperlinks,
        'whitespaces': re_whitespaces,
        'tokenizer': data_tokenizer
    }

    apply_transfomer = Pipeline([
        ('processing_text', ApplyRegex(text_transformers))
    ])

    vectorized_text_pipeline = joblib.load(open(PATH_TO_PIPELINES + "/vectorizer.pkl", "rb"))
    trained_model = joblib.load(open(PATH_TO_MODEL + "/final_model.pkl", "rb"))
    processed_comment = vectorized_text_pipeline.transform(apply_transfomer.transform(data)).toarray()
    return trained_model.predict(processed_comment)

def keyboard_to_url(
    key: str = None,
    key_code: int = None,
    url: str = None,
):
    """Map a keyboard key to open a new tab with a given URL.
    Args:
        key (str, optional): Key to trigger (example 'k'). Defaults to None.
        key_code (int, optional): If key doesn't work, try hard-coding the key_code instead. Defaults to None.
        url (str, optional): Opens the input URL in new tab. Defaults to None.
    """

    assert not (
        key and key_code
    ), """You can not provide key and key_code.
    Either give key and we'll try to find its associated key_code. Or directly
    provide the key_code."""

    assert (key or key_code) and url, """You must provide key or key_code, and a URL"""

    if key:
        key_code_js_row = f"const keyCode = '{key}'.toUpperCase().charCodeAt(0);"
    if key_code:
        key_code_js_row = f"const keyCode = {key_code};"

    components.html(
        f"""
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
{key_code_js_row}
doc.addEventListener('keydown', function(e) {{
    e = e || window.event;
    var target = e.target || e.srcElement;
    // Only trigger the events if they're not happening in an input/textarea/select/button field
    if ( !/INPUT|TEXTAREA|SELECT|BUTTON/.test(target.nodeName) ) {{
        switch (e.keyCode) {{
            case keyCode:
                window.open('{url}', '_blank').focus();
                break;
        }}
    }}
}});
</script>
""",
        height=0,
        width=0,
    )