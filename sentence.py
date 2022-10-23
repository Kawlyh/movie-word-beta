# Download the punkt tokenizer for sentence splitting
# import nltk.data
import wash


import nltk.data
# nltk.download()

# Load the punkt tokenizer model
# this model could be used to tell sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split a review into parsed sentences
def review_to_sentences(review, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    # strip delete string's \n \t \r from head and tail,default
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(wash.review_to_wordlist(raw_sentence, remove_stopwords))
        else:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(wash.review_to_wordlist(raw_sentence))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
