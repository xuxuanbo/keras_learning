# -*- coding:utf-8 -*-
import numpy as np
def get_small_set():
    fs = open('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test_set','r')
    ff = open('/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test_set_1w','a')
    for i in range(10000):
            ff.write(fs.readline())
def analysis_sen_len():
    fs = open('/media/iiip/数据/达官杯数据/new_data/train_set', 'r')
    fs.readline()
    lens = []
    for l in fs.readlines():
        lens.append(len(l.split(',')[1].split(' ')))
    fs.close()
    alens = np.array(lens)
    ranget = np.percentile(alens,95)
    rangeb = np.percentile(alens, 5)
    print ranget,rangeb,np.median(alens)
# analysis_sen_len()
def glove_preprocess():
    fs = open('/media/iiip/数据/达官杯数据/new_data/train_set', 'r')
    ff = open('/media/iiip/数据/达官杯数据/new_data/word.train_set.for_glove', 'a')
    fs.readline()
    for l in fs.readlines():
        ff.write(l.split(',')[2])
    fs.close()
    ff.close()
get_small_set()