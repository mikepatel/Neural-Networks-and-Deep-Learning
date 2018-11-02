# RNN with LSTM units for language modelling
# Text generation: predict next words given a previous history

# dataset: PTB (Penn Tree Bank)
# relatively small and fast to train

# Notes:
#   - http://adventuresinmachinelearning.com/keras-lstm-tutorial/
#   - <eos> = end of sentence characters
#   - Recurrent Batch Normalization (https://arxiv.org/pdf/1603.09025.pdf)
#   - text is already tokenized

################################################################################
# IMPORTs
import os
import numpy as np
import collections
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, BatchNormalization, \
    CuDNNLSTM, LSTM, TimeDistributed
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

################################################################################
# GPU check
GPU = tf.test.is_gpu_available()

################################################################################
# PREPROCESSING
# one-hot encoding of chars/words
# dataset text is already tokenized, so just split sentences


# split sentence tokens
# replace newlines w/ eos
def split_tokens(file):
    with tf.gfile.GFile(file, "r") as f:
        return f.read().replace("\n", "<eos>").split()


# build vocab dictionary
def build_vocab(file):
    tokens = split_tokens(file)

    counter = collections.Counter(tokens)

    # sort by most common char
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# convert text to int
def text_to_ids(file, word_to_id):
    data = split_tokens(file)
    return [word_to_id[word] for word in data if word in word_to_id]


# load dataset
def load_data(file):
    data_filepath = os.path.join(os.getcwd(), "data")
    train_data_filepath = os.path.join(data_filepath, file)

    # build complete vocabulary
    # dictionary w/ key=text, value=int
    # sorted by most common char first
    word_to_id = build_vocab(train_data_filepath)
    #print(word_to_id)

    # convert text -> list of ints
    # train_data will be list of ints
    train_data = text_to_ids(train_data_filepath, word_to_id)

    # limit to most common words (10k)
    vocabulary_size = len(word_to_id)
    #print("VOCAB SIZE: " + str(vocabulary_size))

    # reversed in that key is int and value is text
    # reversed (key, value) pair of word_to_id
    # use predicted int to predict word
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    #print(reversed_dictionary)
    #print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))

    return train_data, vocabulary_size, reversed_dictionary


################################################################################
# HYPERPARAMETERS
NUM_EPOCHS = 50
BATCH_SIZE = 16
DROPOUT_PROB = 0.5
HIDDEN_LAYER = 512
NUM_WORDS = 30


################################################################################
# INPUT PIPELINE FOR FEEDING MODEL
class Generator():
    def __init__(self, data, num_steps, batch_size, vocab_size):
        self.data = data
        # model parameters
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        # counter for iterating
        self.index = 0

    def get_batch(self):
        # yields batches of x and y
        x, y = np.zeros((self.batch_size, self.num_steps)), np.zeros((self.batch_size, self.num_steps, self.vocab_size))
        while True:
            for i in range(self.batch_size):
                # reset index if out of range
                if self.index + self.num_steps >= len(self.data):
                    self.index = 0
                # slice num_steps into each row of x
                x[i, :] = self.data[self.index:self.index + self.num_steps]
                # slice num_steps offet by one and one hot encode y
                y_copy = self.data[self.index + 1:self.index + self.num_steps + 1]
                y[i, :, :] = tf.keras.utils.to_categorical(y_copy, num_classes=self.vocab_size)
                self.index += self.num_steps
            yield x, y


################################################################################
# BUILD AND CONFIGURE MODEL
def build_model(vocabulary_size):
    model = Sequential()

    # Layer 1: Embedding layer
    # convert words to word vectors
    # map integer indices -> dense vectors
    model.add(Embedding(
        input_dim=vocabulary_size,  # size of vocabulary (number of tokens)
        output_dim=HIDDEN_LAYER  # dimension of dense embedding
        #input_length=  # length of input sequence
    ))

    #model.add(BatchNormalization())

    # Layer 2: LSTM
    if GPU:  # use tf.keras.layers.CuDNNLSTM
        model.add(CuDNNLSTM(
            units=HIDDEN_LAYER,
            return_sequences=True  # return all outputs
        ))

    else:  # use tf.keras.layers.LSTM
        model.add(LSTM(
            units=HIDDEN_LAYER,
            return_sequences=True  # return all outputs, also needed to use TimeDistributed wrapper
        ))

    #model.add(BatchNormalization())

    # Layer 3: Dropout
    model.add(Dropout(rate=DROPOUT_PROB))

    #
    if GPU:
        model.add(CuDNNLSTM(
            units=HIDDEN_LAYER,
            return_sequences=True
        ))
    else:
        model.add(LSTM(
            units=HIDDEN_LAYER,
            return_sequences=True
        ))

    model.add(Dropout(rate=DROPOUT_PROB))

    model.add(Dropout(rate=DROPOUT_PROB))

    #
    if GPU:
        model.add(CuDNNLSTM(
            units=HIDDEN_LAYER,
            return_sequences=True
        ))
    else:
        model.add(LSTM(
            units=HIDDEN_LAYER,
            return_sequences=True
        ))

    # Layer 4: Dense
    '''
    # TimeDistributed layer = wrapper that applies a layer to every temporal slice of an input
    # input must be at least 3D, output will be 3D
    model.add(TimeDistributed(
        Dense(
            units=vocabulary_size,
            activation=softmax
        )
    ))
    '''
    model.add(Dense(
        units=vocabulary_size,
        activation=softmax
    ))

    # configure model for training
    # i.e. define loss function, optimizer, training metrics
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["accuracy"]
    )

    model.summary()
    return model


################################################################################
# TRAIN MODEL
def train_model(model, train_data, vocabulary_size, filename):
    # callbacks for saving weights, Tensorboard
    # create a new directory for each run using timestamp
    folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    print("FOLDER: " + folder)
    history_file = str(folder + "\checkpoint_" + filename + ".h5")
    print("HISTORY FILE: " + history_file)
    save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
    tb_callback = TensorBoard(log_dir=folder)

    train_generator = Generator(
        data=train_data,
        num_steps=NUM_WORDS,
        batch_size=BATCH_SIZE,
        vocab_size=vocabulary_size
    )

    history = model.fit_generator(
        generator=train_generator.get_batch(),  # Python iterator used to extract batches
        steps_per_epoch=len(train_data) // (BATCH_SIZE * NUM_WORDS),
        epochs=NUM_EPOCHS,
        callbacks=[save_callback, tb_callback],
        verbose=1
    )

    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]

    return train_accuracy, train_loss, valid_accuracy, valid_loss


################################################################################
#filename = "ptb.char.train.txt"
filename = "ptb.train.txt"

# load dataset
train_data, vocabulary_size, reversed_dictionary = load_data(filename)

# build, train model
model = build_model(vocabulary_size)
train_accuracy, train_loss, valid_accuracy, valid_loss = train_model(model, train_data, vocabulary_size, filename)

################################################################################
# OUTPUT and VISUALIZATION

