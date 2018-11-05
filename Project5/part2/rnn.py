#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rnn.py
# Authors: Michael Patel <mrpatel5@ncsu.edu>, Shahryar Rashid <srashid3@ncsu.edu>, Vijay Mohan <vmohan2@ncsu.edu>

'''
RNN with LSTM Units for Language Modeling
Predicts Next Words Given a History
Dataset: Penn Tree Bank (PTB)
Relatively Small and Fast to Train
Notes
- Recurrent Batch Normalization: https://arxiv.org/pdf/1603.09025.pdf
- Tutorial: http://adventuresinmachinelearning.com/keras-lstm-tutorial/
- <eos> = end of sentence characters
'''

import collections
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, CuDNNLSTM, Dropout, Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

# Check for GPU
GPU = tf.test.is_gpu_available()

# HYPERPARAMETERS and CONSTANTS
C_OR_W = "c"
NUM_EPOCHS = 50
BATCH_SIZE = 32
DROPOUT_PROB = 0.3
HIDDEN_LAYER = 512
NUM_STEPS = 30


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
                y[i, :, :] = to_categorical(y_copy, num_classes=self.vocab_size)
                self.index += self.num_steps
            yield x, y


def load_data(train_filename, val_filename):
    def __split_tokens(file):
        # Replace Newlines with <eos> and Split Tokens
        with open(file, "r") as f:
            return f.read().replace("\n", "<eos>").split()

    def __build_vocab(file):
        # Build Vocabulary Dictionary Sorted by Most Common Char (key=text, value=int)
        tokens = __split_tokens(file)

        counter = collections.Counter(tokens)
        most_frequent = sorted(counter.items(), key=lambda x: -x[1])

        words, frequency = list(zip(*most_frequent))
        return dict(zip(words, range(len(words))))

    def __id_conversion(file, vocab):
        # Convert Text to Int
        tokens = __split_tokens(file)
        return [vocab[token] for token in tokens if token in vocab]

    # filepaths
    data_filepath = os.path.join(os.getcwd(), "data")
    train_filepath = os.path.join(data_filepath, train_filename)
    val_filepath = os.path.join(data_filepath, val_filename)

    vocab = __build_vocab(train_filepath)
    train_data = __id_conversion(train_filepath, vocab)
    val_data = __id_conversion(val_filepath, vocab)

    vocab_size = len(vocab)
    reversed_vocab = dict(zip(vocab.values(), vocab.keys()))

    return train_data, val_data, vocab, vocab_size, reversed_vocab


def build_model(vocab_size):
    # Define Sequential Model (Linear Stack of Layers)
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=HIDDEN_LAYER,
        input_length=NUM_STEPS
    ))

    # LSTM Layer (use CuDNNLSTM for GPUs)
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

    # Dropout Layer
    #model.add(Dropout(rate=DROPOUT_PROB))

    # Dense Layer
    model.add(Dense(
        units=vocab_size,
        activation=softmax
    ))

    # Configure Training (Loss Function, Optimizer, Training Metrics)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_model(model, training_data, val_data, vocab_size, filename):
    # Create Generator for Batches
    train_generator = Generator(
        data=training_data,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size
    )

    val_generator = Generator(
        data=val_data,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size
    )

    # Create Checkpoint
    folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    history_file = str(folder + "\checkpoint_" + filename + ".h5")
    checkpoint = ModelCheckpoint(filepath=history_file, verbose=1)
    tb_callback = TensorBoard(log_dir=folder)

    # Fit Model
    history = model.fit_generator(
        generator=train_generator.get_batch(),
        steps_per_epoch=(len(training_data) // (BATCH_SIZE * NUM_STEPS)),
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint, tb_callback],
        validation_data=val_generator.get_batch(),
        validation_steps=len(val_data) // BATCH_SIZE // NUM_STEPS
    )

    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    validation_accuracy = history_dict["val_acc"]
    validation_loss = history_dict["val_loss"]

    return train_accuracy, train_loss, validation_accuracy, validation_loss, folder, history_file


def generate_sequence(saved_model, vocab, vocab_size, reversed_vocab, length=100):
    model = load_model(saved_model)
    generator = Generator(
        data=training_data,
        num_steps=NUM_STEPS,
        batch_size=1,
        vocab_size=vocab_size
    )

    sequence = ""
    for i in range(length):
        data = next(generator.get_batch())
        output = model.predict(data[0])
        prediction = np.argmax(output[:, NUM_STEPS - 1, :])
        sequence += reversed_vocab[prediction] + " "
    return sequence


##
if C_OR_W == "c":  # character-based
    title = "Character-based"
    train_filename = "ptb.char.train.txt"
    val_filename = "ptb.char.valid.txt"
else:  # word-based
    title = "Word-based"
    train_filename = "ptb.train.txt"
    val_filename = "ptb.valid.txt"

# load dataset
training_data, val_data, vocab, vocab_size, reversed_vocab = load_data(train_filename, val_filename)

# instantiate model
model = build_model(vocab_size)

# train model
train_accuracy, train_loss, val_accuracy, val_loss, folder, history_file = train_model(
    model, training_data, val_data, vocab_size, train_filename
)

# generated text
sequence = generate_sequence(history_file, vocab, vocab_size, reversed_vocab)
print("\n##### Predicted Sequence #####\n")
print(sequence)

# perplexity
train_perplexity = np.exp(train_loss)
val_perplexity = np.exp(val_loss)

print("\n##### Final Perplexity Values: " + title + " #####")
print("Training Perplexity: {}".format(train_perplexity[-1]))
print("Validation Perplexity: {}".format(val_perplexity[-1]))

# plot perplexity
num_epochs_plot = range(1, NUM_EPOCHS+1, 1)
plt.plot(num_epochs_plot, train_perplexity, "b", label="Training Perplexity")
plt.plot(num_epochs_plot, val_perplexity, "r", label="Validation Perplexity")
plt.title("Perplexity: " + title)
plt.xlabel("Number of Epochs")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig(os.path.join(folder, "perplexity_" + title + ".png"))
plt.show()
