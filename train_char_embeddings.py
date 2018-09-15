# _*_ coding: utf-8 _*_

"""
Train distributed representation of characters by using gensim.

Author: StrongXGP
Date:	2018/07/13
"""
#
# import gc
from time import time
import time
from gensim.models.word2vec import Word2Vec


def load_char_samples(train_data_file, test_data_file,val_data_file):
    """Load training and testing data, get the characters of each sample in two dataset and return."""
    train_lines = open(train_data_file, 'r').read().splitlines()[1:]
    test_lines = open(test_data_file, 'r').read().splitlines()[1:]
    # val_lines = open(val_data_file, 'r', encoding='utf-8').read().splitlines()[1:]


    train_char_samples = [" ".join(line.split(',')[2].split()) for line in train_lines]
    test_char_samples = [" ".join(line.split(',')[2].split())for line in test_lines]
    # val_char_samples = [" ".join(line.split()[1:])for line in val_lines]

    char_samples = train_char_samples + test_char_samples #+ val_char_samples

    char_samples = [char_sample.split() for char_sample in char_samples]

    del train_lines, test_lines, train_char_samples, test_char_samples#,val_char_samples
    # gc.collect()

    return char_samples


def batch_iter(data, batch_size=5000):
    """Generate batch iterator."""
    data_size = len(data)
    num_batches = ((data_size - 1) // batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def main():
    # Load data
    # =========================================================================

    print("[INFO] Loading data...")
    train_data_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/train_set"
    test_data_file = "/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/test_set"
    val_data_file = './new_data/data_daguan_val_all.txt'
    sentences = load_char_samples(train_data_file, test_data_file,val_data_file)
    print("[INFO] The total number of samples is: %d" % len(sentences))

    # Calculate the size of vocabulary
    # =========================================================================

    # chars = []
    # for sentence in sentences:
    #     chars.extend(sentence)
    # print("[INFO] The total number of characters is: %d" % len(set(chars)))
    #
    # del chars
    # gc.collect()

    # Train and save word2vec model
    # =========================================================================

    print("[INFO] Initialize word2vec model...")
    model = Word2Vec(size=50, min_count=5, sg=0, iter=30, workers=16, seed=42)
    model.build_vocab(sentences)
    # print("[INFO] ", end='')
    print(model)

    print("[INFO] Start training...")
    t0 = time.time()
    batches = batch_iter(sentences, batch_size=1000)
    for batch in batches:
        model.train(batch, total_examples=len(batch), epochs=model.epochs)
    print("[INFO] Done in %.3f seconds!" % (time.time() - t0))
    print("[INFO] Training finished! ( ^ _ ^ ) V")

    print("[INFO] Save to file...")
    model.wv.save("/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/datagrand-char-50d.bin")
    model.wv.save_word2vec_format("/media/hadoopnew/08D40FCFD40FBE46/达官杯数据/new_data/datagrand-char-50d.txt", binary=False)
    print("[INFO] Finished!")


if __name__ == '__main__':
    main()
