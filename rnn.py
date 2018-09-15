# -*- coding:utf-8 -*-
import keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.layers import GlobalMaxPooling1D
from data_preprocess import word_cnn_train_batch_generator
from sklearn.model_selection import train_test_split
import sklearn
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import layers, models, optimizers
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils import np_utils
def load_data_and_embedding(path):
    MAX_SEQUENCE_LENGTH = 1200  # 每个文本的最长选取长度，较短的文本可以设短些
    EMBEDDING_DIM = 300  # 词向量的维度，可以根据实际情况使用，如果不了解暂时不要改
    VALIDATION_SPLIT = 0.4  # 这里用作是测试集的比例，单词本身的意思是验证集
    MAX_NB_WORDS = 300000  # 整体词库字典中，词的多少，可以略微调大或调小
    # first, build index mapping words in the embeddings set
    # to their embe dding vector  这段话是指建立一个词到词向量之间的索引，比如 peking 对应的词向量可能是（0.1,0,32,...0.35,0.5)等等。
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/embedding/datagrand-char-300d.txt')  # 读入50维的词向量文件，可以改成100维或者其他
    f.readline()
    for line in f:
        # print line
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')  # 下面这段代码，主要作用是读入训练样本，并读入相应的标签，并给每个出现过的单词赋一个编号，比如单词peking对应编号100
    texts = []  # 存储训练样本的list
    label = []
    fs = open(path)
    fs.readline()
    for line in fs.readlines():
        texts.append(line.split(',')[2])
        label.append(line.split(',')[3])

    print('Found %s texts.' % len(texts))  # 输出训练样本的数量

    # finally, vectorize the text samples into a 2D integer tensor,下面这段代码主要是将文本转换成文本序列，比如 文本'我爱中华' 转化为[‘我爱’，'中华']，然后再将其转化为[101,231],最后将这些编号展开成词向量，这样每个文本就是一个2维矩阵，这块可以参加本文‘二.卷积神经网络与词向量的结合’这一章节的讲述
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(label))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set,下面这段代码，主要是将数据集分为，训练集和测试集（英文原意是验证集，但是我略有改动代码）
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]  # 训练集
    y_train = labels[:-nb_validation_samples]  # 训练集的标签
    x_val = data[-nb_validation_samples:]  # 测试集，英文原意是验证集
    y_val = labels[-nb_validation_samples:]  # 测试集的标签

    # prepare embedding matrix 这部分主要是创建一个词向量矩阵，使每个词都有其对应的词向量相对应
    # nb_words = min(MAX_NB_WORDS, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        # if i > MAX_NB_WORDS:注释后不设词数限制
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector  # word_index to word_embedding_vector ,<300000(nb_words)

    # load pre-trained word embeddings into an Embedding layer
    # 神经网路的第一层，词向量层，本文使用了预训练glove词向量，可以把trainable那里设为False
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                dropout=0.2,
                                trainable=True)

    print('Training model.')

    # train a 1D convnet with global maxpoolinnb_wordsg
    print('Preparing embedding matrix.')

    del data, labels,label, texts,

    return x_train,y_train,x_val ,y_val,embedding_layer

def get_word_lstm(word_embed_layer):
    # model = Sequential()
    # model.add(word_embed_layer)
    # model.add(Dropout(0.2))
    # model.add(LSTM(150))
    # model.add(Dense(256,activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(20,activation='softmax'))
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def get_word_bilstm( word_embed_layer):

    model = Sequential()
    model.add(word_embed_layer)
    model.add(Bidirectional(LSTM(150, return_sequences=True),merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(20,activation="softmax"))
    # returns a sequence of vectors of dimension 32

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print _val_f1
        return
def get_word_rnn(word_len, word_embed_layer):
    word_input = Input(shape=(word_len,), dtype="int32")
    # char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = word_embed_layer
    # char_embedding = Embedding(
    #     name="char_embedding",
    #     input_dim =char_embed_weight.shape[0],
    #     weights =[char_embed_weight],
    #     output_dim=char_embed_weight.shape[1],
    #     trainable=False
    # )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    # trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    # word_bigru2 = Bidirectional(GRU(128, return_sequences=True))(word_bigru)
    avg_pool = GlobalAveragePooling1D()(word_bigru)
    max_pool = GlobalMaxPooling1D()(word_bigru)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    # char_bigru2 = Bidirectional(GRU(128, return_sequences=True))(char_bigru)
    # avg_pool = GlobalAveragePooling1D()(char_bigru)
    # max_pool = GlobalMaxPooling1D()(char_bigru)
    # char_feat = concatenate([avg_pool, max_pool], axis=1)
    # feat = concatenate([word_feat], axis=1)
    dropfeat = Dropout(0.3)(word_feat) # 0.4
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat))) # 256
    output = Dense(20, activation="softmax")(fc)
    # model = Model(inputs=[word_input, char_input], outputs=output)
    model = Model(inputs=word_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


x_train, y_train, x_val, y_val,embedding_layer = load_data_and_embedding('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/train_set_1w')
model = get_word_rnn(1200,embedding_layer)
batch_size = 128
metrics = Metrics()
for i in range(4):
    print('---------------------EPOCH------------------------')
    print(i)
    model.fit_generator(
        word_cnn_train_batch_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train.shape[0] / batch_size),
        #callbacks=[metrics],
        validation_data=(x_val, y_val)
    )

# model.fit(x_train, y_train,
#           batch_size=64,
#           epochs=100,
#           validation_data=(x_val, y_val))
# score = model.evaluate(x_val, y_val,
#                        batch_size=128
#                        )
    model.save('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/model/my_model3.h5')
# print score