# -*- coding:utf-8 -*-
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
import sklearn
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import Concatenate
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.utils import np_utils
from keras import layers

MAX_SEQUENCE_LENGTH = 3000 # 每个文本的最长选取长度，较短的文本可以设短些
EMBEDDING_DIM = 50 # 词向量的维度，可以根据实际情况使用，如果不了解暂时不要改
VALIDATION_SPLIT = 0.4 # 这里用作是测试集的比例，单词本身的意思是验证集
MAX_NB_WORDS = 300000 # 整体词库字典中，词的多少，可以略微调大或调小

def load_train_set_data(path):

    text = []
    label = []
    # print text,'\n',label
    fs = open(path)
    fs.readline()
    for line in fs.readlines():
        text.append(line.split(',')[2])
        label.append(line.split(',')[3])

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.1, random_state=42)

    #格式转换
    y_test=pd.Series(y_test)
    y_train=pd.Series(y_train)
    # 对类别变量进行编码，共10类
    y_labels = list(y_train.value_counts().index)
    # print y_labels
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)


    # 分词，构建单词-id词典
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(text)
    vocab = tokenizer.word_index
    #
    # # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    print X_train_word_ids.__len__()
    # One-hot
    # x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
    # x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')
    # lens_text = map(lambda i:len(i.split(" ")),text)
    # MAX_SEQUENCE_LENGTH = max(lens_text)
    # print x_train
    # 序列模式
    x_train = pad_sequences(X_train_word_ids, maxlen=3000)
    x_test = pad_sequences(X_test_word_ids, maxlen=3000)
    #
    return x_train, x_test, y_train, y_test,3000,vocab.__len__()


def load_d2v_embedding(path,dim):
    id_cate_vec = np.loadtxt(path, dtype=str, delimiter='\t')
    train_set = id_cate_vec[:, 2]
    label = id_cate_vec[:, 1]
    # 划分训练/测试集
    x_train, x_test, y_train, y_test = train_test_split(train_set, label, test_size=0.1, random_state=42)
    # print x_train, x_test, '\n',y_train, y_test
    x_train = np.array([map(lambda x: float(x), i.split()) for i in x_train], dtype=float)
    # print x_train.shape
    x_test = np.array([map(lambda x: float(x), i.split()) for i in x_test], dtype=float)

    # # 对类别变量进行编码，共10类
    # y_labels = list(y_train.value_counts().index)
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(label)
    num_labels = 19
    y_train = to_categorical(map(lambda x: le.transform([x])[0], y_train), num_labels)
    # print y_train
    y_test = to_categorical(map(lambda x: le.transform([x])[0], y_test), num_labels)
    return x_train, x_test, y_train, y_test,dim


def cnn_text(path):
    # first, build index mapping words in the embeddings set
    # to their embedding vector  这段话是指建立一个词到词向量之间的索引，比如 peking 对应的词向量可能是（0.1,0,32,...0.35,0.5)等等。
    print('Indexing word vectors.')

    embeddings_index = {}

    f = open('../new_data/datagrand-char-50d.txt')# 读入50维的词向量文件，可以改成100维或者其他
    f.readline()
    for line in f:
        #print line
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
    #nb_words = min(MAX_NB_WORDS, len(word_index))
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
                                trainable=True)

    print('Training model.')

    # train a 1D convnet with global maxpoolinnb_wordsg
    print('Preparing embedding matrix.')
    model_left = Sequential()
    # model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_left.add(embedding_layer)

    # left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）
    model_left.add(Conv1D(128, 5, activation='tanh'))
    model_left.add(MaxPooling1D(5))
    model_left.add(Conv1D(128, 5, activation='tanh'))
    model_left.add(MaxPooling1D(5))
    model_left.add(Conv1D(128, 5, activation='tanh'))
    model_left.add(MaxPooling1D(35))
    model_left.add(Flatten())
    model_left.add(Dropout(0.1))
    model_left.add(BatchNormalization())  # (批)规范化层

    # right model 第二块神经网络，卷积窗口是4*50

    model_right = Sequential()
    model_right.add(embedding_layer)
    model_right.add(Conv1D(128, 4, activation='tanh'))
    model_right.add(MaxPooling1D(4))
    model_right.add(Conv1D(128, 4, activation='tanh'))
    model_right.add(MaxPooling1D(4))
    model_right.add(Conv1D(128, 4, activation='tanh'))
    model_right.add(MaxPooling1D(28))
    model_right.add(Flatten())
    model_right.add(Dropout(0.1))
    model_right.add(BatchNormalization())  # (批)规范化层

    # third model 第三块神经网络，卷积窗口是6*50
    model_3 = Sequential()
    model_3.add(embedding_layer)
    model_3.add(Conv1D(128, 6, activation='tanh'))
    model_3.add(MaxPooling1D(3))
    model_3.add(Conv1D(128, 6, activation='tanh'))
    model_3.add(MaxPooling1D(3))
    model_3.add(Conv1D(128, 6, activation='tanh'))
    model_3.add(MaxPooling1D(30))
    model_3.add(Flatten())
    model_3.add(Dropout(0.1))
    model_3.add(BatchNormalization())  # (批)规范化层


    #merged = merge([model_left, model_right, model_3],mode='concat')
    # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
    #merged = Concatenate(-1)([model_left.output, model_right.output,model_3.output])
    #print merged.shape()
    merged = layers.Merge([model_left, model_right, model_3],mode='concat',concat_axis=1)
    model = Sequential()
    model.add(merged)  # add merge
    model.add(Dense(128, activation='tanh'))  # 全连接层
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='softmax'))  # softmax，输出文本属于19种类别中每个类别的概率

    # 优化器我这里用了adadelta，也可以使用其他方法
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])

    # =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
    model.fit(x_train, y_train, nb_epoch=6,batch_size=320,validation_data=(x_val, y_val))

    score = model.evaluate(x_train, y_train, verbose=0)  # 评估模型在训练集中的效果，准确率约99%
    print('train score:', score[0])
    print('train accuracy:', score[1])
    # score = model.evaluate(x_val, y_val, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

def single_model(path):
    print('Indexing word vectors.')

    embeddings_index = {}

    f = open('../new_data/datagrand-char-50d.txt')  # 读入50维的词向量文件，可以改成100维或者其他
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
                                trainable=True)

    print('Training model.')

    # train a 1D convnet with global maxpoolinnb_wordsg
    print('Preparing embedding matrix.')
    model = Sequential()
    # model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model.add(embedding_layer)
    # 模型结构：嵌入-卷积池化*2-dropout-BN-全连接-dropout-全连接
    # model.add(Embedding(len + 1, 300))
    model.add(Convolution1D(256, 3, padding='same'))  # ,input_shape=(x_train.shape[1],x_train.shape[2])
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(128, 3, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=320,
              epochs=100,
              validation_data=(x_val, y_val))

# x_train, x_test, y_train, y_test,dim = load_d2v_embedding('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test_d2v',19)
# x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])

single_model('/media/iiip/数据/达官杯数据/new_data/train_set')

