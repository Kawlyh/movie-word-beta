from gensim.models import Word2Vec
from joblib.numpy_pickle_utils import xrange
from sklearn.cluster import KMeans
import time
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bag_of_centroids import create_bag_of_centroids

# set log
from process_ave_vec import clean_train_reviews, clean_test_reviews

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, )
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3, )
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, )
print("Read %d labeled train reviews, %d labeled test reviews,and %d unlabeled reviews\n" % (
    train["review"].size, test["review"].size, unlabeled_train["review"].size))

# Load the model that we created in process_word2vec
model = Word2Vec.load("300features_40minwords_10context")

# set start time,because the time of running a k_means process is very long
# we want to see the time
start = time.time()

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.vectors
num_clusters = word_vectors.shape[0] // 5

# Initialize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word dictionary, mapping each vocabulary word to a cluster number
word_centroid_map = dict(zip(model.index_to_word, idx))

# test look
# For the first 10 clusters
for cluster in xrange(0, 10):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all the words for that cluster number, and print them out
    words = []
    for i in xrange(0, len(word_centroid_map.values())):
        if word_centroid_map.values()[i] == cluster:
            words.append(word_centroid_map.keys()[i])
    print(words)

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators=100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
