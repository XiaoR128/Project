import numpy as np
import jieba.posseg
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim.models as g
from gensim.models.word2vec import LineSentence
import re

#


#读取数据
def read(file):
    f=open(file,'r',encoding='utf-8')
    lines=f.readlines()
    li=[s.rstrip('\n') for s in lines]
    # print(li)
    return li

# 读取触发词
def read_ewords(file):
    ty=[]
    words=[]
    with open(file,encoding='utf-8') as fr:
        for line in fr:
            typ=line.split(':')
            ty.append(typ[0])
            ww=typ[1].split(' ')
            words.append(ww)
    return ty,words

# 提取包含触发词的句子
def takesent(newstr,words,length):
    li = [[] for i in range(0, length)]
    for s in newstr:
        for i in range(0, length):
            for word in words[i]:
                if word in s:
                    li[i].append(s)
    return li


#获取包含触发词的字符串
def takestr(file):
    ty, words = read_ewords('eventwords.txt')
    length = len(ty)
    newstr=read(file)
    li = takesent(newstr, words, length)

    # 所有类别的list集合
    totallist = []

    # totallist.append(li[0])
    # totallist.append(li[1])
    # totallist.append(li[2])
    # totallist.append(li[3])

    # print(totallist)

    totallist.append(list(set(li[0])))
    totallist.append(list(set(li[1])))
    totallist.append(list(set(li[2])))
    totallist.append(list(set(li[3])))

    print(len(totallist[0]))
    print(len(totallist[1]))
    print(len(totallist[2]))
    print(len(totallist[3]))

    writeback(totallist)
    writeback_details(totallist)
    return totallist

#将找到的结果写回文件
def writeback(stoplist):
    f=open('tfidfdata_new.txt','w',encoding='utf-8')
    i=0
    for list in stoplist:
        for s in list:
            f.write(s+' ')
        f.write('\n')
    f.close()

def writeback_details(stoplist):
    f = open('tfidfdata_new_det.txt', 'w', encoding='utf-8')
    i = 0
    for list in stoplist:
        for s in list:
            f.write(str(i)+'\t'+s)
            f.write('\n')
        i+=1
    f.close()

#读取停顿词
def readstop():
    f=open('stop_words_zh.txt','r',encoding='utf-8')
    li=f.readlines()
    stopwords=[s.rstrip('\n') for s in li]
    return stopwords

#对数据进行分词和去除停顿词
def datapro(text,stopwords):

    li=[]
    length=len(text)
    #判断是否是中文字符
    cn_reg = '^[\u4e00-\u9fa5]+$'

    for i in range(0,length):
        cut_res = jieba.cut(text[i])
        res = []
        for s in cut_res:
            if s not in stopwords and re.search(cn_reg, s):
                res.append(s)
        li.append(res)

    return li

#找出topk的词
def tfidf_topk(li,topk):
    #将分词用空格隔开并全部放在一行中
    corpus=[]
    for i in range(0,len(li)):
        temp=' '.join(li[i])
        corpus.append(temp)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vect = CountVectorizer()
    X = vect.fit_transform(corpus)  # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vect.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    tfidf_weight = tfidf.toarray()

    keys=[]
    for i in range(len(tfidf_weight)):
        df_word, df_weight = [], []
        for j in range(len(word)):
            # print('词'+word[j]+'的权值为'+str(tfidf_weight[i][j]))
            df_word.append(word[j])
            df_weight.append(tfidf_weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1)  # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by='weight', ascending=False)  # 按照权重值降序排列
        keyword = np.array(word_weight['word'])  # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0, topk)]  # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        keys.append(word_split)
    print(keys)

    return keys

#训练词向量
def word2vecTrain(text):
    model = g.Word2Vec(LineSentence(text), size=100, window=5, min_count=1)
    model.save('Word2VecModel.model')
    model.wv.save_word2vec_format('Word2VecModel.vector', binary=False)

#清理标注后的数据
def dealnodata():
    f = open('tfidfdata_new_det.txt', 'r', encoding='utf-8')
    li = f.readlines()
    f.close()
    words = [s.rstrip('\n') for s in li]

    fs = open('filtdata.txt', 'r', encoding='utf-8')
    lis = fs.readlines()
    fs.close()
    new_words = [s.rstrip('\n') for s in lis]

    # wbs = open('tfidfdata_new.txt', 'w', encoding='utf-8')
    #
    #
    # data=[]
    # labels=[]
    # for s in words:
    #     la=s.split('\t')
    #     labels.append(int(la[0]))
    #     data.append(la[1])
    #
    # leng=len(labels)
    # count=0
    # for i in range(leng):
    #     if labels[i]==count:
    #         wbs.write(data[i]+' ')
    #     else:
    #         count+=1
    #         wbs.write('\n'+data[i]+' ')
    #
    # wbs.close()



    noevdata=[]
    for sw in new_words:
        if sw not in data:
            noevdata.append(sw)

    print(len(noevdata))

    wb = open('noeventdata.txt', 'w', encoding='utf-8')
    for s in noevdata:
        wb.write(str(4)+'\t'+s)
        wb.write('\n')
    wb.close()




#使用模型
def usemodel():
    # fv=fast2vec()
    model = g.KeyedVectors.load_word2vec_format('zhwiki_2017_03.sg_50d.word2vec',binary=False)
    model.init_sims(replace=True)

    type1 = [u'好吃', u'今天', u'吃饭', u'感觉', u'火锅', u'美食']
    list2 = [u'烧烤', u'鲍鱼', u'面条']

    ww=model.n_similarity(type1,list2)
    print(ww)

    #
    # type1 = [u'好吃', u'一个', u'今天', u'吃饭' ,u'没有' ,u'真的' ,u'喜欢' ,u'我们' ,u'一起' u'火锅']
    # type2 = [u'一个', u'难受', u'没有', u'真的', u'感冒', u'今天', u'心疼', u'现在', u'痛苦', u'感觉']
    # type3 = [u'旅行', u'一个', u'喜欢', u'风景', u'我们', u'颜色', u'今天', u'真的', u'黑色', u'说走就走']
    # type4 = [u'一个', u'我们', u'学校', u'上', u'真的', u'学会', u'同学', u'喜欢', u'今天' u'考试']
    #
    # list2 = list(jieba.cut('我们很开心'))
    # s = model.wv.n_similarity(type1, list2)
    # print(s)


def main():
    # totallist=takestr('filtdata.txt')

    #对所有的语料库进行分词相关处理
    # wholetext=read('filtdata.txt')
    # stopwords = readstop()
    # res=datapro(wholetext,stopwords)
    # f = open('wordtraindata.txt', 'w', encoding='utf-8')
    # for line in res:
    #     if len(line):
    #         p = ' '.join(line)
    #         f.write(p + '\n')
    # f.close()


    # dealnodata()

    #利用tf-idf找到文档中的topk的词
    text=read('tfidfdata_new.txt')
    stopwords=readstop()
    li=datapro(text,stopwords)
    tfidf_topk(li,10)

    #训练数据
    # word2vecTrain('wordtraindata.txt')

    #利用训练得到的数据
    # usemodel()

    #test




if __name__ == '__main__':
    main()


