import pandas as pd
import jieba
import matplotlib.pyplot as plt
import datetime
import sys
from gensim.models import word2vec

def dealdata(list):
    numsten = [str(s).strip() for s in list]
    news = [k for k in numsten if k != '']
    nnews = [t.replace(' 我在:', '') for t in news]
    r1=[t.replace(' 我在这里:', '') for t in nnews]
    return r1

def jiebawords(newstr):
    seg=[]
    for i in range(len(newstr)):
        seg.append([' '.join(list(jieba.cut(newstr[i],cut_all=False)))])
    return seg

def write_seg(seg):
    with open('seg_words.txt','w',encoding='utf-8') as f:
        for k in seg:
            f.writelines(k)
            f.write('\n')
    f.close()

#训练词向量
def train_w2v(filename):
    text=word2vec.LineSentence(filename)
    model=word2vec.Word2Vec(text,sg=1,hs=1,min_count=1,window=5,size=300)
    model.save('./mymodel')

def load_model(filename):
    model=word2vec.Word2Vec.load(filename)

    # print('火锅和面条相似度',model.similarity('火锅','面条'))

    sim1=model.most_similar('难受',topn=10)
    for key in sim1:
        print('和难受有关的词有',key[0],',距离是',key[1])

if __name__ == '__main__':
    data = pd.read_csv('w1.csv', engine='python')
    fr = data['content']
    fr=fr.dropna()

    stes=fr.tolist()
    newstr=dealdata(stes)
    seg=jiebawords(newstr)
    print(seg)
    # write_seg(seg)
    # train_w2v('seg_words.txt')
    # load_model('./mymodel')

    # i=0
    # for st in newstr:
    #     print(st)
    #     i+=1
    # print(i)

    # word1='旅游'
    # for sentence in fr:
    #     sen=str(sentence)
    #     c=sen.count(word1)
    #     if c>0:
    #         print(sen)
