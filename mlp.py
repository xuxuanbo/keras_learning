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
import sklearn
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.utils import np_utils
from data_preprocess import batch_generator, get_word_seq
import pickle
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import random
def tfidf_generator(features, labels, batch_size):

    # Create empty arrays to contain batch of features and labels#

    batch_features = np.zeros((batch_size,features.shape[1]))

    batch_labels = np.zeros((batch_size,1))

    while True:

        for i in range(batch_size):

        # choose random index in features

            index= random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]

            yield batch_features, batch_labels

def data_generator(data, targets, batch_size):
    batches = (len(data) + batch_size - 1)//batch_size
    while(True):
         for i in range(batches):
              X = data[i*batch_size : (i+1)*batch_size]
              Y = targets[i*batch_size : (i+1)*batch_size]
              yield (X, Y)
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(metrics.val_f1s)
        return
metrics = Metrics()

f=open('/media/iiip/数据/达官杯数据/new_data/tfidf/data_tfidf_select_LSVC_l2_901288.pkl','rb')
train_set,label,test_set = pickle.load(f)
f.close()
x_train, x_test, y_train, y_test = train_test_split(train_set, label, test_size=0.1, random_state=42)
y_train = keras.utils.to_categorical(y_train, num_classes=19)
y_test = keras.utils.to_categorical(y_test, num_classes=19)
num_labels = 19

model = Sequential()
# 全连接层
model.add(Dense(512, input_shape=(x_train.shape[1],), activation='relu'))
# DropOut层
model.add(Dropout(0.5))
# # 全连接层+分类器
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# # model.add(Dense(64,activation='relu'))
# # model.add(Dropout(0.5))
model.add(Dense(19,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=320,
#           epochs=100,
#           validation_data=(x_test, y_test),callbacks=[metrics])
batch_size = 320
# model.fit_generator(generator = data_generator(x_train, y_train, batch_size=320),
#                     steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size,
#                     epochs = 100,
#                     verbose = 1,
#                     callbacks = [metrics],
#                     validation_data=(x_test, y_test)
# )
model.fit_generator(
    tfidf_generator(x_train, y_train, batch_size=batch_size),
    epochs=1,
    steps_per_epoch=int(x_train.shape[0] / batch_size),
    callbacks=[metrics],
    validation_data=(x_test, y_test)
)
# score = model.evaluate(x_test, y_test,
#                        batch_size=320
#                        )
pred = np.squeeze(model.predict(test_set))
f=open('/media/iiip/数据/达官杯数据/new_data/tfidf/data_tfidf_predict.pkl','w')
pickle.dump(pred,f)
f.close()
# model.save('my_model3.h5')
# print score