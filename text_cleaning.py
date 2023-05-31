import pandas as pd
import spacy  # for getting a list of stopwords
from pymorphy2 import MorphAnalyzer  # for lemmatization
import warnings
import re


def clean_and_tokenize(text):
    """
    cleans text from bad symbols
    """
    tokenized = re.findall(r'[а-я]+', text, flags=re.IGNORECASE)
    return [i.lower() for i in tokenized]


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
    nlp = spacy.load("ru_core_news_sm")
    stopwords = nlp.Defaults.stop_words
    nlp.Defaults.stop_words |= {'нрзб', 'го', 'г', 'н'}
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


def remove_non_rus(dataframe, header):
    for text_id, text in enumerate(dataframe[header]):
        if all(symbol.lower() not in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя123456789' for symbol in text):
            dataframe.loc[text_id, header] = ''


def preprocess(text):
    """
    unite clean, tokenize, remove_stopwords, and lemmatize functions
    returns a string!
    """
    tokens = clean_and_tokenize(text)
    cleaned_tokens = remove_stopwords(tokens)
    lemmas = lemmatize(cleaned_tokens)
    lemmas = ' '.join(lemmas)
    return lemmas


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # opening our corpus
    df = pd.read_excel('cleaned_corpus.xlsx')

    # replace text with preprocessed text
    df['Текст открытки'] = df['Текст открытки'].apply(preprocess)

    # removes postcards with foreign texts
    remove_non_rus(df, 'Текст открытки')

    # finds all rows without text and deletes them
    condition1 = (df['Текст открытки'].apply(tokenize).apply(len) == 0)
    condition2 = (df['Текст открытки'].apply(tokenize).apply(len) == 1)
    df.drop(df[condition1].index, inplace=True)
    df.drop(df[condition2].index, inplace=True)

    # save the corpus with lemmatized texts to a new xlsx document
    df.to_excel('lemmatized_texts.xlsx', index=False)
