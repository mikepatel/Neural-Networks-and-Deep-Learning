#

################################################################################
# IMPORTs
import os
import numpy as np
import collections
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, CuDNNLSTM, LSTM
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


################################################################################
# HYPERPARAMETERS and CONSTANTS
# check if using GPU
GPU = tf.test.is_gpu_available()

NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_STEPS = 50  # number of words fed into dense layer, set of words model will
                # learn from in order to predict words after
DROPOUT_RATE = 0.5
NUM_NODES_HIDDEN_LAYER = 512


################################################################################
# fn to read files
def split_tokens(file):
    with tf.gfile.GFile(file, "r") as f:
        return f.read().replace("\n", "<eos>").split()


# fn that creates reverse dictionary, vectorized training data
def load_data(filename):
    data_filepath = os.path.join(os.getcwd(), "data")
    train_data_filepath = os.path.join(data_filepath, filename)

    # create list of words (aka tokens)
    tokens_list = split_tokens(train_data_filepath)

    # create dict w/ key=words, value=count
    counter = collections.Counter(tokens_list)

    # convert to list of pairs (word, count), sorted by count
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # create list of words sorted by most common first
    words, _ = list(zip(*count_pairs))

    # create dict w/ key=words, value=index, sorted by most common first
    # all words are mapped to a unique int
    # 10k most common
    word_dict = dict(zip(words, range(len(words))))

    # 10k words
    size_vocab = len(word_dict)

    # create a reversed dict w/ key=index, value=word
    # given index, can find corresponding word needed to construct text output
    reversed_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

    # vectorization of training data text file
    # list of index
    vector_train_data = [word_dict[word] for word in tokens_list if word in word_dict]

    return vector_train_data, size_vocab, reversed_word_dict


################################################################################
# fn that builds and configures model
def build_model(size_of_vocab):
    m = Sequential()

    #
    m.add(Embedding(
        input_dim=size_of_vocab,
        output_dim=NUM_NODES_HIDDEN_LAYER
    ))

    #
    if GPU:
        m.add(CuDNNLSTM(
            units=NUM_NODES_HIDDEN_LAYER,
            return_sequences=True
        ))
    else:
        m.add(LSTM(
            units=NUM_NODES_HIDDEN_LAYER,
            return_sequences=True
        ))

    #
    m.add(Dropout(rate=DROPOUT_RATE))

    #
    m.add(Dense(
        units=size_of_vocab,
        activation=softmax
    ))

    # configure model for training
    # i.e. define loss fn, optimizer, metrics
    m.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["accuracy"]
    )

    m.summary()
    return m


################################################################################
#
filename = "ptb.train.txt"
vector_train_data, size_vocab, reversed_word_dict = load_data(filename)

# instantiate model
model = build_model(size_vocab)

# callbacks
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
history_file = str(folder + "\checkpoint_" + filename + ".h5")
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)

# train model
history = model.fit(
    x=vector_train_data,
    y=,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[save_callback, tb_callback],
    verbose=1
)

history_dict = history.history
train_accuracy = history_dict["acc"]
train_loss = history["loss"]

print("\n##########")
print("Loss: {}".format(train_loss))
print("Accuracy: {}".format(train_accuracy))
print("\n##########")