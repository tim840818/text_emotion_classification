import re
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def clean_text(text):
    text = re.sub(r"[^\w\s']",'',text, re.UNICODE)
    text = text.lower()
    return text

def remove_stopwords(text, do_stem=False):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    white_list = ['not', 'no', 'nor']
    white_list += [word for word in stop_words if "n't" in word]
    black_list = ["i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll", "i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd"]
    stop_words -= set(white_list)
    stop_words |= set(black_list)
    tokens = []
    for token in text.split():
        # if token not in stop_words:
        #     if do_stem:
        #         tokens.append(stemmer.stem(token))
        #     else:
        #         tokens.append(token)
        if do_stem: # Stemming first and then removing stopwords
            token = stemmer.stem(token)
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)

def preprocess_text(text, do_stem=False):
    # text = text.lower()
    text = clean_text(text)
    text = remove_stopwords(text, do_stem)
    return text

def text2sentences(text):
    return [sentence.split() for sentence in text.tolist()]

def get_embeddingdict_w2v(tokenizer, model_w2v):
    num_words = len(tokenizer.word_index) + 1

    embedding_matrix = np.zeros((num_words, model_w2v.vector_size))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model_w2v.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            pass

    embedding_dict = {
    'input_dim': num_words,
    'output_dim': model_w2v.vector_size,
    'weights': [embedding_matrix],
    }
    return embedding_dict

def get_embeddingdict_glove(tokenizer, path_to_glove_file):
    num_words = len(tokenizer.word_index) + 1
    embedding_index = {}
    hits = 0
    misses = 0
    # Read word vectors
    with open(path_to_glove_file) as f:
        for line in f:
            word, coeffs = line.split(maxsplit=1)
            coeffs = np.fromstring(coeffs, "f", sep=" ")
            embedding_index[word] = coeffs
    print("Found %s word vectors." % len(embedding_index))
    embedding_dim = len(coeffs)

    # Assign word vectors to our dictionary/vocabulary
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else: # Words not found in embedding index will be all-zeros.
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    embedding_dict = {
        'input_dim': num_words,
        'output_dim': embedding_dim,
        'weights': [embedding_matrix],
    }
    return embedding_dict
