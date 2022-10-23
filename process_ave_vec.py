# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word removal.
import pandas as pd
from gensim.models import Word2Vec
import logging
from makefearue import getAvgFeatureVecs
from wash import review_to_wordlist
from process_word2vec import num_features
from sklearn.ensemble import RandomForestClassifier

# set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the model that we created in process_word2vec
model = Word2Vec.load("300features_40minwords_10context")

# read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, )
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3, )
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, )
print("Read %d labeled train reviews, %d labeled test reviews,and %d unlabeled reviews\n" % (
    train["review"].size, test["review"].size, unlabeled_train["review"].size))

# Calculate average feature vectors for training and testing sets,
# Notice that we now use stop word
print("Creating average feature vecs for train reviews")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
