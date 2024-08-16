from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Masking, LSTM, Bidirectional, Dropout
from tensorflow.keras import optimizers
import keras_tuner as kt


def nlp_dnn(input_shape, categories, compile=False):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape)) # (train_X.shape[1],)
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(categories, activation='softmax'))

    if compile: model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'f1_score'])
    return model

def nlp_lstm_embedding(embedding_dict, categories, input_length=None, compile=False, verbose=0):
    # if input_length: # this parameter sets the padding size directly so we don't need to construct embedding_dict all over again
    #     embedding_dict['input_length'] = input_length
    if verbose >= 1:
        print("Training data size: ", embedding_dict['input_dim'])
        print("Embedding dimension: ", embedding_dict['output_dim'])
        # print("Padding size: ", embedding_dict['input_length'])
    if verbose == 2:
        print("Embedding matrix shape: ", embedding_dict['weights'][0].shape)
    
    model = Sequential()
    model.add(Embedding(**embedding_dict, trainable=False))
    model.add(Masking(mask_value=0.))
    model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=False, dropout=0.2)))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(categories, activation='softmax'))

    if compile: model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'f1_score'])
    return model

## Hyperparameter model
def nlp_hyperparam(hp:kt.HyperParameters, embedding_dict=None, input_length=None, verbose=0):
    if not embedding_dict:
        raise ValueError("embedding_dict is required")
    # if input_length: # this parameter sets the padding size directly so we don't need to construct embedding_dict all over again
    #     embedding_dict['input_length'] = input_length
    if verbose:
        print("Training data size: ", embedding_dict['input_dim'])
        print("Embedding dimension: ", embedding_dict['output_dim'])
        # print("Padding size: ", embedding_dict['input_length'])

    model_type = hp.Choice('model_type', values=['single_LSTM', 'bidir_LSTM'])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    dropout_rate = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)

    model = Sequential()
    model.add(Embedding(**embedding_dict, trainable=False))
    model.add(Masking(mask_value=0.))
    if model_type == 'single_LSTM':
        model.add(LSTM(hp.Int('units', min_value=32, max_value=64, step=16), activation='tanh', dropout=dropout_rate, return_sequences=True))
        model.add(LSTM(hp.Int('units', min_value=16, max_value=32, step=8), activation='tanh', dropout=dropout_rate, return_sequences=False))
    elif model_type == 'bidir_LSTM':
        model.add(Bidirectional(LSTM(hp.Int('units', min_value=32, max_value=64, step=16), activation='tanh', dropout=dropout_rate, return_sequences=True)))
        model.add(Bidirectional(LSTM(hp.Int('units', min_value=16, max_value=32, step=8), activation='tanh', dropout=dropout_rate, return_sequences=False)))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'f1_score'])
    return model
