import os

import gc

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import sklearn
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
#from init.config import Config
from data_preprocess import word_cnn_train_batch_generator, get_word_seq
from keras.utils.np_utils import to_categorical
from deepzoo import get_rcnn
from my_utils.metrics import score

print("Load train @ test")
maxlen = 2000

print('Loading data...')
def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('/media/iiip/软件/daguan/new_data/train_ids_and_labels.txt')
    y = df_data['class'] - 1  # class (0 ~ 18)
    X = df_data.drop(['class'], axis=1).values

    # Transform to binary class matrix
    y = to_categorical(y.values)

    # Randomly shuffle data
    np.random.seed(10)

    shuffle_indices = np.random.permutation(range(len(y)))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    del df_data, X, y, X_shuffled, y_shuffled

    embedding_matrix = np.load("/media/iiip/软件/daguan/embedding/word-embedding-300d-mc5.npy")

    return X_train, y_train, X_val, y_val,embedding_matrix



x_train,y_train, x_val,  y_val, embedding_matrix= load_data_and_embedding()


# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
load_val = True
batch_size = 128
model_name = "word_rcnn"
trainable_layer = ["embedding"]
train_batch_generator = word_cnn_train_batch_generator

print("Load Word")

model = get_rcnn(maxlen, embedding_matrix,  embedding_matrix.shape[0]+1)

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file= model_name+ '.png',show_shapes=True)

best_f1 = 0
for i in range(25):
    print('---------------------EPOCH------------------------')
    print(i)
    if best_f1 > 0.73:
        K.set_value(model.optimizer.lr, 0.0001)
    if best_f1 > 0.73:
        for l in trainable_layer:

            model.get_layer(l).trainable = True

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train.shape[0] / batch_size),
        validation_data=(x_val, y_val)
    )

    if best_f1 > 0.72:
        pred = np.squeeze(model.predict(x_val))
        pre, rec, f1 = score(pred, y_val)
        print("precision", pre)
        # print("recall", rec)
        print("f1_score", f1)

    if (best_f1 > 0.72 and float(f1) > best_f1):
        print('saving model (｡・`ω´･) ')
        best_f1 = f1
        model.save('/media/iiip/软件/daguan/save_model/' + "dp_embed_%s.h5" % (model_name))

    # model.save(Config.cache_dir + '/rcnn/dp_embed_%s_epoch_%s_%s.h5'%(model_name, i, f1))