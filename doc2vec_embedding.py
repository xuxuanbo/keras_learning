# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import gensim, logging
import csv
import numpy as np
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import gensim.models.doc2vec
from  gensim.models.doc2vec  import  Doc2Vec , TaggedDocument
import random
def init():
    # logging information
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # get input file, text format

    csvFile = '/media/hadoopnew/CB1D-DA82/new_data/train_set.csv'
    csv.field_size_limit(sys.maxsize)
    with open(csvFile,'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]
    print rows[0]['word_seg']
    documents = [TaggedDocument(doc['word_seg'].split(), [i]) for i, doc in enumerate(rows)]
    model = Doc2Vec(documents, vector_size=128, window=8, min_count=1, workers=5)
    path = './my_model'
    model.save(path)
    return path

def d2v(model_path,train_set_path,train_word_set_path):

    model = gensim.models.Doc2Vec.load(model_path)
    print len(model.docvecs)

def txt_2_npy(path):
    id_cate_vec = np.loadtxt(path, dtype=str,delimiter='\t')
    # print id_cate_vec
    train_set = id_cate_vec[:, 2]
    label = id_cate_vec[:, 1]
    x_train = np.array([map(lambda x: float(x), i.split()) for i in train_set], dtype=float)
    label = np.array([map(lambda x: float(x), i.split()) for i in label], dtype=float)
    matrix = np.hstack((x_train,label))
    np.save('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test', matrix)

asin=set()
class LabeledLineSentence(object):
    def __init__(self, filename=object):
        self.filename =filename
    def __iter__(self):
        with open(self.filename,'r') as infile:
            infile.readline()
            data=infile.readlines()
            print "length: ", len(data)
        for uid,line in enumerate(data):
            # print "line:",line
            asin.add(line.split(",")[0]+'#'+ line.split(",")[3].strip())
            # print "asin: ",asin
            yield LabeledSentence(words=line.split(",")[2].split(" "), tags=[line.split(",")[0]+'#'+ line.split(",")[3].strip()])

# logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
# sentences =LabeledLineSentence('/media/hadoopnew/CB1D-DA82/new_data/train_set')
# model = Doc2Vec(vector_size=256, window=8, min_count=1, workers=5)
# senLst = [sent for sent in sentences]
# model.build_vocab(senLst)
#
# # train the model
# for epoch in range(5):
#     logging.info('epoch %d' % epoch)
#     random.shuffle(senLst)
#     model.train(senLst,
#                 total_examples=model.corpus_count,
#                 epochs=model.iter
#                 )
# model.save('myd2vmodel2')
# print  'success1'
#
# print "doc2vecs length:", len(model.docvecs)
# outid = file('/media/hadoopnew/CB1D-DA82/new_data/train_set_id_vector3.txt', 'a')
# for id in asin:
#     outid.write('\t'.join(id.split('#'))+"\t")
#     for idx,lv in enumerate(model.docvecs[id]):
#         outid.write(str(lv)+" ")
#     outid.write("\n")
#
# outid.close()









txt_2_npy('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/train_set_id_vector1024.txt')