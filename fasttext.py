#!/usr/bin/env python
# coding=utf-8

import codecs
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import logging
import sklearn
from data_preprocess import word_cnn_train_batch_generator
from keras.models import Model
import re
import pickle as pkl

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s: %(message)s', datefmt='%Y-%m-%d %H:%M',
#                     filename='log/train_model.log', filemode='a+')

ngram_range = 1
#max_features = 6500
#maxlen = 120

# fw = open('error_line_test.txt', 'wb')
#
# DIRTY_LABEL = re.compile('\W+')
# # set([u'业务',u'代销',u'施工',u'策划',u'设计',u'销售',u'除外',u'零售',u'食品'])
# STOP_WORDS = pkl.load(open('./data/stopwords.pkl'))
def load_whole_set(path):
    texts = []  # 存储训练样本的list
    text_set = []
    fs = open('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test_set')
    fs.readline()
    for line in fs.readlines():
        texts.append(line.split(',')[2])
    fs = open('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/train_set')
    fs.readline()
    for line in fs.readlines():
        text_set.append(line.split(',')[2])
    print('Found %s texts.' % len(texts))  # 输出训练样本的数量

    # finally, vectorize the text samples into a 2D integer tensor,下面这段代码主要是将文本转换成文本序列，比如 文本'我爱中华' 转化为[‘我爱’，'中华']，然后再将其转化为[101,231],最后将这些编号展开成词向量，这样每个文本就是一个2维矩阵，这块可以参加本文‘二.卷积神经网络与词向量的结合’这一章节的讲述
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(text_set)
    sequences = tokenizer.texts_to_sequences(texts)

    text = pad_sequences(sequences, maxlen=1600)
    return text,1600
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
    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.001, random_state=42)

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
    # 序列模式
    x_train = pad_sequences(X_train_word_ids, maxlen=1600)
    x_test = pad_sequences(X_test_word_ids, maxlen=1600)
    return x_train, x_test, y_train, y_test,1600,vocab.__len__()

def load_data(fname='data/12315_industry_business_train.csv', nrows=None):
    """
    载入训练数据
    """
    data, labels = [], []
    char2idx = json.load(open('data/char2idx.json'))
    used_keys = set(['name', 'business'])
    df = pd.read_csv(fname, encoding='utf-8', nrows=nrows)
    for idx, item in df.iterrows():
        item = item.to_dict()
        line = ''
        for key, value in item.iteritems():
            if key in used_keys:
                line += key + value

        data.append([char2idx[char] for char in line if char in char2idx])
        labels.append(item['label'])

    le = LabelEncoder()
    logging.info('%d nb_class: %s' % (len(np.unique(labels)), str(np.unique(labels))))
    onehot_label = to_categorical(le.fit_transform(labels))
    joblib.dump(le, 'model/tgind_labelencoder.h5')
    x_train, x_test, y_train, y_test = train_test_split(data, onehot_label, test_size=0.1)
    return (x_train, y_train), (x_test, y_test)


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of sequences by appending n-grams values

    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences



def cal_accuracy(x_test, y_test):
    """
    准确率统计
    """
    y_test = np.argmax(y_test, axis=1)
    y_pred = model.predict_classes(x_test)
    correct_cnt = np.sum(y_pred == y_test)
    return float(correct_cnt) / len(y_test)

def middle_layer_output(model_path):
    model = load_model(model_path)
    #dense1_layer_model = Model(inputs=model.input,
    #                           outputs=model.get_layer('GAP').output)

    # 以这个model的预测值作为输出

    text, maxlen = load_whole_set('test_set')
    output = model.predict(text)
    # output = np.argmax(output, axis=0)
    #print output.shape
    #dense1_output = dense1_layer_model.predict(text)
    np.save('fasttext_test',output)
    # print dense1_output.shape
def ngram(x_train, x_test, y_train, y_test, maxlen, vocab_len):


    logging.info('x_train size: %d' % (len(x_train)))
    logging.info('x_test size: %d' % (len(x_test)))
    # logging.info('x_train sent average len: %.2f' % (np.mean(list(map(len, x_train)))))
    # print 'x_train sent avg length: %.2f' % (np.mean(list(map(len, x_train))))

    if ngram_range > 1:
        print 'add {}-gram features'.format(ngram_range)
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        start_index = vocab_len + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        max_features = np.max(list(indice_token.keys())) + 1

        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
    return x_train, x_test
    print 'pad sequences (samples x time)'
    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')


def run(x_train, x_test, y_train, y_test, vocab_len,nb_class):
    logging.info('x_train.shape: %s' % (str(x_train.shape)))

    print 'build model...'

    DEBUG = True
    if DEBUG:
        model = Sequential()
        print vocab_len
        model.add(Embedding(vocab_len+1, 200, input_length=1600))
        model.add(GlobalAveragePooling1D(name='GAP'))
        model.add(Dropout(0.3))
        model.add(Dense(nb_class, activation='softmax'))
    else:
        model = load_model('./model/tgind_dalei.h5')

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # earlystop = EarlyStopping(monitor='val_loss', patience=8)
    # checkpoint = ModelCheckpoint(filepath='./model/tgind_dalei.h5', monitor='val_loss', save_best_only=True,
    #                              save_weights_only=False)
    for i in range(13):
        print('---------------------EPOCH------------------------')
        print(i)
        batch_size = 64
        #model.fit(x_train, y_train, shuffle=True, batch_size=64, epochs=80, validation_split=0.1, )
        model.fit_generator(
            word_cnn_train_batch_generator(x_train, y_train, batch_size=batch_size),
            epochs=1,
            steps_per_epoch=int(x_train.shape[0] / batch_size),
            #callbacks=[metrics],
            validation_data=(x_test, y_test)
        )
        model.save('fasttext_test.h5')

          #callbacks=[checkpoint, earlystop])


    # loss, acc = model.evaluate(x_test, y_test)
    # print '\n\nlast model: loss', loss
    # print 'acc', acc
    #
    # model = load_model('model/tgind_dalei.h5')
    # loss, acc = model.evaluate(x_test, y_test)
    # print '\n\n cur best model: loss', loss
    # print 'accuracy', acc
    # logging.info('loss: %.4f ;accuracy: %.4f' % (loss, acc))
    #
    # logging.info('\nmodel acc: %.4f' % acc)
    # logging.info('\nmodel config:\n %s' % model.get_config())
# x_train, x_test, y_train, y_test, maxlen, vocab_len = load_train_set_data('train_set')
# nb_class = 19
# run(x_train, x_test, y_train, y_test, vocab_len,nb_class)
middle_layer_output('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/new_model/fasttext_test.h5')