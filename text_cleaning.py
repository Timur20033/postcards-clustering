import pandas as pd
import nltk  # for getting a list of stopwords
from pymorphy2 import MorphAnalyzer  # for lemmatization


def clean(text):
    """
    cleans text from bad symbols
    """
    cleaned_text = ''
    for symbol in text:
        if symbol.isalnum() or symbol == ' ':
            cleaned_text += symbol
        else:
            cleaned_text += ' '
    return cleaned_text


def tokenize(text):
    """
    tokenize text
    """
    tokens = text.split()
    clean_tokens = [token.strip().lower() for token in tokens]
    return clean_tokens


def remove_stopwords(tokens):
    """
    removes stopwords from tokens sequence
    """
    stopwords = nltk.corpus.stopwords.words('russian')
    stopwords.append('нрзб')
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords:
            cleaned_tokens.append(token)
    return cleaned_tokens


def lemmatize(tokens):
    """
    makes lemmas from tokens sequence
    """
    morph = MorphAnalyzer()
    lemmas = []
    for token in tokens:
        lemmas.append(morph.normal_forms(token)[0])
    return lemmas


def preprocess(text):
    """
    unite clean, tokenize, remove_stopwords, and lemmatize functions
    returns a string!
    """
    cleaned_text = clean(text)
    tokens = tokenize(cleaned_text)
    cleaned_tokens = remove_stopwords(tokens)
    lemmas = lemmatize(cleaned_tokens)
    lemmas = ' '.join(lemmas)
    return lemmas


# opening our corpus
df = pd.read_excel('cleaned_corpus.xlsx')

# extract texts to a list
texts = []
for txt in df['Текст открытки']:
    texts.append(preprocess(txt))

# replace texts in the corpus by preprocessed texts
for txt in df["Текст открытки"]:
    for i in range(74):
        df.loc[i, 'Текст открытки'] = texts[i]

# save the corpus with lemmatized texts to a new xlsx document
# df.to_excel('lemmatized_texts.xlsx', index=False)


