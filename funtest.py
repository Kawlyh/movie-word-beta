from gensim.models import Word2Vec
import logging
# set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Load the model that we created in process
model = Word2Vec.load("300features_40minwords_10context")
# see vectors
print(model.wv.vectors.shape)
print(type(model.wv.vectors))
print(model.wv["flower"])
print(model.wv.most_similar("flower"))