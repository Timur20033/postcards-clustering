import warnings

# to transfrom our corpus to a tf-idf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import pandas as pd

# to scale data
from sklearn.preprocessing import MinMaxScaler

# methods of clustering
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# to visualize the n-dimensional tf-idf matrix in 2-dimensional plot
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

# for defining topics
from gensim import corpora, models


warnings.filterwarnings("ignore")

# reading the corpus from the document
df = pd.read_excel('lemmatized_texts.xlsx')

# extract texts to a list to give it to Tfidfvectorizer as an argument
texts = [str(text) for text in df['Текст открытки']]

# making a vectorizer
vectorizer = TfidfVectorizer()

# fit_transfrom method learns all our corpus and transfrom it to the tf-idf matrix
matrix = vectorizer.fit_transform(texts)

# km_scores = []
# for i in range(2, 12):
#     km = KMeans(n_clusters=i, random_state=0).fit(matrix)
#     preds = km.predict(matrix)
#
#     print("Score for number of cluster(s) {}: {}".format(i, km.score(matrix)))
#     km_scores.append(-km.score(matrix))

# plt.figure(figsize=(7, 4))
# plt.title("The elbow method", fontsize=16)
# plt.scatter(x=[i for i in range(2, 12)], y=km_scores, s=150, edgecolor='k')
# plt.grid(True)
# plt.xlabel("Number of clusters", fontsize=14)
# plt.ylabel("K-means score", fontsize=15)
# plt.xticks([i for i in range(2, 12)], fontsize=14)
# plt.yticks(fontsize=15)


# using K-Means clustering method
kmeans = KMeans(n_clusters=2, random_state=42)  # initializing a model1
kmeans.fit(matrix)  # teaching the model
clusters = kmeans.labels_  # a list of clusters of the same length as the number of texts in the corpus

# using the MiniBatchKMeans method
# mbk = MiniBatchKMeans(init='random', n_clusters=3)
# mbk.fit_transform(matrix)
# clusters = mbk.labels_.tolist()

# using the DBSCAN method
# db = DBSCAN()
# db.fit(matrix)
# clusters = db.labels_

# using the Agglomeration method
# agglo1 = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
# answer = agglo1.fit_predict(matrix.toarray())
# clusters = answer.tolist()

# for visualizing transfroming our n-dimensional matrix to the 2-dimentional one
pca = PCA(n_components=2, random_state=42)  # n-components represent the number of dimensions needed
pca_vecs = pca.fit_transform(matrix.toarray())  # making the 2-dimensional matrix

# this is two basis vectors for our 2-dimensional matrix
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# saving clusters and basis vectors as a new columns in our corpus
df['Кластер'] = clusters
df['x0'] = x0
df['x1'] = x1

# just visualizing clusters
# plt.figure(figsize=(12, 7))
# plt.title("clustering", fontdict={"fontsize": 18})
# plt.xlabel("X0", fontdict={"fontsize": 16})
# plt.ylabel("X1", fontdict={"fontsize": 16})
# sns.scatterplot(data=df, x='x0', y='x1', hue='Кластер', palette="viridis")
# plt.show()

list_of_list_of_tokens = [text.split(' ') for text in df['Текст открытки']]

dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

num_topics = 2
lda_model = models.EnsembleLda(corpus,
                            num_topics=num_topics,
                            id2word=dictionary_LDA,
                            passes=4,
                            alpha=[0.01]*num_topics,
                            eta=[0.01]*len(dictionary_LDA.keys())
                            )

for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(i, topic)
    print()

# saving a the clustered corpus to a new xlsx document
# cluster_df = df[['Номер открытки', 'Текст открытки', 'Кластер']].copy()
# cluster_df.to_excel('clustered_corpus.xlsx', index=False)




