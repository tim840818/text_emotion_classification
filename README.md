# Text Emotion Classification
This project analyzes and classifies the emotions from the given text dataset with two different methods: "Bag of words + DNNs model" and "Padding sequences + Word-to-Vector + LSTMs model". Please refer to [`text_classify.ipynb`](./text_classify.ipynb) for the demonstration.


## Data sources
The dataset comes from Praveen's [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) on Kaggle. The dataset contains 16000 training rows, 2000 validating rows, and 2000 testing rows. Each row is a sentence corresponding to one of the six emotions.

In addition, a pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) model is used for word-to-vector implementation.


## Methodology
The dataset is imported and analyzed with `pandas`. Two emotions "love" and "surprise" are dropped out due to lack of data. The new dataset for deep learning is narrowwed down to 14124 training rows, 1741 validating rows, and 1775 testing rows, corresponding to four total emotions.

Before training, the data is preprocessed by removing punctuation and stop words and stemming the remaining words.

Two methods shown below are developed to train the dataset.

### Bag of words + DNN model
The training corpus is tokenized into 9531 unique tokens (words). Each data row can be represented as a 9531-dimension vector representing the occurrence numbers of tokens. A Deep Neural Network (DNN) is constructed with two Dense layers and one output layer.

### Padding sequences + Word-to-Vector + LSTM model
Instead of bag of words, each data row is now padded into a sequence of tokens with a specific parameter `pad_size`. Each token is represented as a 50-dimension vector, and an embedding matrix in the shape of 9531 x 50 is built. Each row is processed through an embedding layer and embedded into an array with the shape (`pad_size`, 50). It is then passed through a masking layer, two LSTM layers, and an output layer for training. Finally, we tune this model using Hyperband.


## Results

|          | Bag-of-Words + DNN | GloVe + LSTM |
|----------|--------------------|--------------|
| Accuracy | 0.89               | 0.82         |

For the bag of words method, we obtain 0.89 accuracy. The accuracy is fairly good, but one of the disadvantages is obvious that it requires too many spaces and is too sparse.

For the padding sequence method with pretrained GloVe word-to-vector model, our model achieves 0.82 accuracy. The input size of each training data is 13 x 50 = 650. Although the accuracy is not better than one from the bag of words method, it requires much less space.

## Modules used
* `pandas`: Constructs a datatable to organize and manipulate all data.
* `sklearn`: Uses LabelEncoder to encode emotions to numbers.
* `nltk`: NLP package for removing stop words and stemming.
* `tensorflow.keras`: Tokenizes words, pads sentences into sequences, and constructs DNN and LSTM models.
* `keras_tuner`: Tunes hyperparameters.

## Programs included
* [`text_classify.ipynb`](./text_classify.ipynb):
    * [`text_preprocess.py`](text_preprocess.py): Preprocesses texts by removing stop words and stemming; constructs embedding matrix from word-to-vector models.
    * [`nlp_model.py`](nlp_model.py): Constructs DNN models, LSTM models, and hyperparameter tuning LSTM models.
