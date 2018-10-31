# RNN with LSTM units for language modelling
# Text generation: predict next words given a previous history

# dataset: PTB (Penn Tree Bank)
# relatively small and fast to train

# Notes:
#   - http://adventuresinmachinelearning.com/keras-lstm-tutorial/
#   - <eos> = end of sentence characters
#   - Recurrent Batch Normalization (https://arxiv.org/pdf/1603.09025.pdf)

################################################################################
# IMPORTs
import os
import collections
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, BatchNormalization, \
    CuDNNLSTM, LSTM, TimeDistributed
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

tf.enable_eager_execution()

################################################################################
# GPU check
is_use_GPU = tf.test.is_gpu_available()

################################################################################
# PREPROCESSING


# split words
def read_words(file):
    with tf.gfile.GFile(file, "r") as f:
        return f.read().replace("\n", "<eos>").split()


# build vocab dictionary
def build_vocab(file):
    data = read_words(file)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# convert text to int
def file_to_word_ids(file, word_to_id):
    data = read_words(file)
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
    train_data = file_to_word_ids(train_data_filepath, word_to_id)

    # limit to most common words (10k)
    vocabulary_size = len(word_to_id)
    # print(vocabulary)

    # reversed in that key is int and value is text
    # reversed (key, value) pair of word_to_id
    # use predicted int to predict word
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    #print(reversed_dictionary)
    #print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))

    return train_data, vocabulary_size, reversed_dictionary


################################################################################
# HYPERPARAMETERS
NUM_EPOCHS =
BATCH_SIZE =
DROPOUT_PROB = 0.5  # vary from 0.5 to 0.9

################################################################################
# INPUT PIPELINE FOR FEEDING MODEL

################################################################################
# EMBEDDING
# convert words to vectors
# convert unique words to unique integer index


################################################################################
# BUILD AND CONFIGURE MODEL
def build_model(vocabulary_size):
    model = Sequential()

    # Layer 1: Embedding layer
    # convert words to vectors
    model.add(Embedding(
        input_dim=vocabulary_size,  # size of vocabulary
        output_dim=,  # dimension of dense embedding
        input_length=  # length of input sequence
    ))

    #model.add(BatchNormalization())

    # Layer 2: LSTM
    if is_use_GPU:  # use tf.keras.layers.CuDNNLSTM
        model.add(CuDNNLSTM(
            units=,
            return_sequences=True  # return all outputs
        ))

    else:  # use tf.keras.layers.LSTM
        model.add(LSTM(
            units=,
            return_sequences=True  # return all outputs, also needed to use TimeDistributed wrapper
        ))

    #model.add(BatchNormalization())

    # Layer 3: Dropout
    model.add(Dropout(rate=DROPOUT_PROB))

    # Layer 4: Dense
    # TimeDistributed layer = wrapper that applies a layer to every temporal slice of an input
    # input must be at least 3D, output will be 3D
    model.add(TimeDistributed(
        Dense(
            units=vocabulary_size,
            activation=softmax
        )
    ))

    # configure model for training
    # i.e. define loss function, optimizer, training metrics
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["categorical_accuracy"]
    )

    model.summary()
    return model


################################################################################
# TRAIN MODEL
def train_model(model):
    history = model.fit_generator(

    )

################################################################################
# OUTPUT

################################################################################
filename = "ptb.char.train.txt"
#filename = "ptb.train.txt"
load_data(filename)

# build, train model

# callbacks for saving weights, Tensorboard
# create a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
history_file = folder + ".h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)