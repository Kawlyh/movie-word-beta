# word2vec deeplearning don't need labels
# 单词向量模型，本模型为所有单词创建一个特征向量
# 利用特征向量进行聚类等操作
import pandas as pd
import sentence
import logging
from gensim.models import word2vec


# read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3,)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3,)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3,)
print("Read %d labeled train reviews, %d labeled test reviews,and %d unlabeled reviews\n" % (
    train["review"].size, test["review"].size, unlabeled_train["review"].size))

# parse review data into list of word list,and wash them
sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += sentence.review_to_sentences(review)
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += sentence.review_to_sentences(review)

# half test
print(len(sentences))
print(sentences[0])
print(sentences[1])

# train
#
# set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words
print("Training model...")
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          vector_size=num_features,# 输出词向量的维数，即神经网络的隐藏层的单元数
                          min_count=min_word_count,# 频率小于40的的单词会被忽视
                          window=context,# 句子中当前词与目标词之间的最大距离，即为窗口，设置窗口移动的大小为5
                          sample=downsampling)# sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

# final test
print(model.wv.similar_by_word("awful"))