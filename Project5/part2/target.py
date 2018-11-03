import os
import tensorflow as tf
import collections


def split_tokens(file):
    with tf.gfile.GFile(file, "r") as f:
        return f.read().replace("\n", "<eos>").split()


filename = "ptb.train.txt"
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
print(vector_train_data[:105])

dataX = []
dataY = []
seq_length = 100

for i in range(0, len(tokens_list) - seq_length):
    seq_in = tokens_list[i: i+seq_length]
    seq_out = tokens_list[i+seq_length]
    dataX.append([word_dict[c] for c in seq_in])
    dataY.append(word_dict[seq_out])

print("\n")
print(dataX[0])
print("\n")
print(dataY[:5])