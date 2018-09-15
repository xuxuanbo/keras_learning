# _*_ coding: utf-8 _*_

"""
Re-save the file of word embeddings to `npy` format and
construct the mapping between words and its corresponding indices.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/29
"""

import pickle
import numpy as np
import time

EMBEDDING_SIZE = 50
SPECIAL_SYMBOLS = ['<PAD>', '<UNK>']

# Load words and its corresponding embeddings
# ===========================================================================================

print("Load words and its corresponding embeddings...")
np.random.seed(42)
word_embedding_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/datagrand-char-50d.txt"
with open(word_embedding_file, 'r') as f:
    lines = f.read().splitlines()[1:]

    word_to_id_map = dict()
    id_to_word_map = dict()
    for i, symbol in enumerate(SPECIAL_SYMBOLS):
        id_to_word_map[i] = symbol
        word_to_id_map[symbol] = i

    num_total_symbols = len(lines) + len(SPECIAL_SYMBOLS)
    word_embeddings = np.zeros((num_total_symbols, EMBEDDING_SIZE), dtype=np.float32)
    word_embeddings[1] = np.random.randn(EMBEDDING_SIZE)  # the values of 'UNK' satisfy the normal distribution

    index = 2
    for line in lines:
        cols = line.split()
        id_to_word_map[index] = cols[0]
        word_to_id_map[cols[0]] = index
        word_embeddings[index] = np.array(cols[1:], dtype=np.float32)
        index += 1

# Save to file
# ===========================================================================================

print("Save to file...")
id2word_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/embedding/id2word.pkl"
word2id_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/embedding/word2id.pkl"
word_embeddings_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/embedding/word-embedding-50d-mc5.npy"
with open(id2word_file, 'wb') as fout:
    pickle.dump(id_to_word_map, fout)
with open(word2id_file, 'wb') as fout:
    pickle.dump(word_to_id_map, fout)
np.save(word_embeddings_file, word_embeddings)
print("Finished! ( ^ _ ^ ) V")
