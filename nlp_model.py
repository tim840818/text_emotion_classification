from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Masking, LSTM, Bidirectional, Dropout
from tensorflow.keras import optimizers
import keras_tuner as kt

import torch


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

## training / evaluation models for torch
def train_model(model, train_loader, device, epoch, epochs=12, optimizer=None, lr_scheduler=None):
    from tqdm.auto import tqdm
    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    train_progress_bar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch + 1}/{epochs} ")
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step() if lr_scheduler is not None else None

        train_loss += loss.item()

        # progress_bar.update(1)
        train_progress_bar.update(1)
        avg_loss = train_loss / (batch_idx + 1)
        train_progress_bar.set_postfix({"loss": avg_loss})
    
    train_progress_bar.close()


def evaluate_model(model, data_loader, device, tqdm_=False, mode='val', out=True):
    total_val_loss = 0
    correct = 0
    total = 0
    model.eval()
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(**batch)

        val_loss = outputs.loss
        logits = outputs.logits

        total_val_loss += val_loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_val_loss = total_val_loss / len(data_loader)
    val_accuracy = correct / total

    val_word = 'Validation' if 'val' in mode else 'Test'

    if tqdm_:
        from tqdm.auto import tqdm
        tqdm.write(f"{val_word} Loss: {avg_val_loss:.4f}, {val_word} Accuracy: {val_accuracy:.4f}")
    else:
        print(f"{val_word} Loss: {avg_val_loss:.4f}, {val_word} Accuracy: {val_accuracy:.4f}")

    if out:
        return avg_val_loss, val_accuracy

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if (validation_loss - self.min_validation_loss) < -self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
        return False