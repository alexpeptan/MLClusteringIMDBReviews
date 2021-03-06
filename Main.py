# Load Training Data
with open("resources/imdb_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')

print("Loaded " + str(len(lines)) + " comments from IMDB.")

# Separate Problem Instance(comment) from its label(0(Negative)/1(Positive)) and filter out incorrect data
splitLines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']

train_documents = [line[0] for line in splitLines]

# Use TfidfVectorizer to represent comments numerically using Term Frequency - Inverse Document Frequency Representation
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
train_documents_tfidf = tfidf_vectorizer.fit_transform(train_documents)

print("\nFor first comment: ")
print(splitLines[0])

print("\nWeighted computed term frequencies by TF-IDF: ")
print(train_documents_tfidf[0])

print("\nWeights of all the words that still remained in the dictionary after fit_transform on TF-IDF: ")
print(tfidf_vectorizer.vocabulary_)

# Perform Clustering
from sklearn.cluster import KMeans

clusters_nr = 3
print("\nClustering comments into " + str(clusters_nr) + " clusters:")
km = KMeans(n_clusters=clusters_nr, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(train_documents_tfidf)


def print_k_comments_from_cluster_c(comments_nr, cluster_nr):
    print("\nFirst " + str(comments_nr) + " comments from cluster " + str(cluster_nr) + ": ")
    count = 0
    for i in range(len(splitLines)):
        if count >= comments_nr:
            break
        if km.labels_[i] == cluster_nr:
            print(splitLines[i])
            count += 1


print_k_comments_from_cluster_c(10, 0)
print_k_comments_from_cluster_c(10, 1)
print_k_comments_from_cluster_c(10, 2)
