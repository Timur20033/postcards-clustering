import pandas as pd

# reading an initial corpus from the xlsx document
corpus = pd.read_excel('corpus.xlsx')

# replace NaN by empty strings
corpus = corpus.fillna('')

# replace [отсутствует] by empty strings
corpus = corpus.replace('[отсутствует]', '')

# filter the corpus by postcards where content is not an empty string
corpus = corpus[corpus['Текст открытки'] != '']

# filter our corpus by postcards which do not have any of all seven tags
tags = [1, 2, 3, 4, 5, 6, 7]
for tag in tags:
    corpus = corpus[corpus[f'Тег_{tag}'] == '']

# saving necessary columns of the filtered corpus to a new corpus
cleaned_corpus = corpus[['Номер открытки', 'Текст открытки']].copy()

# saving the new corpus to an xlsx document
#cleaned_corpus.to_excel('cleaned_corpus.xlsx', index=False)
