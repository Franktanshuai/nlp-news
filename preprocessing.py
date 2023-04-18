import pandas as pd
import re
import jieba
import gensim
import pickle
import numpy as np


def main():
    training_set_path = "dataset/train.csv"
    test_set_path = "dataset/test.csv"
    training_set = dataset2vec(training_set_path, is_training_set=True)
    test_set = dataset2vec(test_set_path, is_training_set=False)

    with open("dataset/training_set.pkl", "wb") as file:
        pickle.dump(training_set, file)
    with open("dataset/test_set.pkl", "wb") as file:
        pickle.dump(test_set, file)


def dataset2vec(file_path, is_training_set):
    df = pd.read_csv(file_path)

    df.drop(axis=1, inplace=True, columns=["Unnamed: 0"])
    df = df.replace(re.compile(r'\[.*?\]'), " ", regex=True)  # [xxx] 表情符号
    df = df.replace(re.compile(r'@.*?:'), " ", regex=True)  # @xxx
    df = df.replace("\t", " ", regex=False)  # 转移符号
    df = df.replace("网页链接", " ", regex=False)  # 链接
    df['content'] = df['content'].str.strip()
    df = df.fillna(value=' ')  # 补充空值
    df['content'] = df['content'].apply(lambda x: ' '.join(sentence_segment(x)))
    df['comment_all'] = df['comment_all'].apply(lambda x: ' '.join(sentence_segment(x)))
    df = df.fillna(value=' ')

    content_seg_list = [x.split(' ') for x in df['content']]
    comment_seg_list = [x.split(' ') for x in df['comment_all']]
    wv_model = None
    wv_size = 50
    if is_training_set:
        wv_model = gensim.models.word2vec.Word2Vec(content_seg_list + comment_seg_list, vector_size=wv_size, min_count=1)
        with open('model/wv.model', 'wb') as outfile:
            pickle.dump(wv_model, outfile)
    else:
        with open('model/wv.model', 'rb') as infile:
            wv_model = pickle.load(infile)

    feature = []
    for i in range(len(content_seg_list)):
        feature_vec = np.zeros(shape=[0], dtype='float32')
        text_vec = np.zeros(shape=[wv_size], dtype='float32')
        count = 0
        for word in content_seg_list[i]:
            if wv_model.wv.has_index_for(word):
                text_vec += wv_model.wv[word]
                count += 1
        if count != 0:
            feature_vec = np.concatenate((feature_vec, text_vec / count))
        else:
            feature_vec = np.concatenate((feature_vec, text_vec))

        text_vec = np.zeros(shape=[wv_size], dtype='float32')
        count = 0
        for word in comment_seg_list[i]:
            if wv_model.wv.has_index_for(word):
                text_vec += wv_model.wv[word]
                count += 1
        if count != 0:
            feature_vec = np.concatenate((feature_vec, text_vec / count))
        else:
            feature_vec = np.concatenate((feature_vec, text_vec))
        feature.append(feature_vec.tolist())

    label = []
    for x in df['label']:
        label.append(x)
    return {'X': np.array(feature), 'y': np.array(label)}


def sentence_segment(sentence):
    pattern = re.compile("[^\u4e00-\u9fa5]+")
    sentence = pattern.sub('', sentence)
    stopwords_list = pd.read_table('dataset/cn_stopwords.txt', header=None).iloc[:, :].values
    seg_list = jieba.cut(sentence)
    filtered_words_list = []
    for word in seg_list:
        if word not in stopwords_list:
            filtered_words_list.append(word)
    return filtered_words_list


main()
