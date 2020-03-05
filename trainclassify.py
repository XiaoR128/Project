import gensim.models as g
import jieba.posseg
import jieba.analyse
import re
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import *
import numpy as np

#读取需要训练的数据
def readtrain(file):
    f=open(file,'r',encoding='utf-8')
    lines=f.readlines()
    newline=[s.rstrip('\n') for s in lines]
    labels=[]
    contents=[]
    for s in newline:
        sp=s.split('\t')
        labels.append(sp[0])
        contents.append(sp[1])
    # print(labels)
    # print(contents)
    return labels,contents

#读取停顿词
def readstop():
    f=open('stop_words_zh.txt','r',encoding='utf-8')
    li=f.readlines()
    stopwords=[s.rstrip('\n') for s in li]
    return stopwords

#对数据进行分词等处理
def cutwords(text,labels):
    stopwords=readstop()
    li=[]
    length=len(text)
    #判断是否是中文字符
    cn_reg = '^[\u4e00-\u9fa5]+$'
    lab=[]

    for i in range(0,length):
        cut_res = jieba.cut(text[i])
        res = []
        for s in cut_res:
            if s not in stopwords and re.search(cn_reg, s):
                res.append(s)
        if len(res):
            li.append(res)
            lab.append(labels[i])


    return li,lab

#计算训练数据的词向量和之前得到的词向量之间的相似度
def calsimilar(li):
    model = g.KeyedVectors.load_word2vec_format('zhwiki_2017_03.sg_50d.word2vec', binary=False)
    model.init_sims(replace=True)

    # 通过tf-idf得到的词结果
    type1 = [u'好吃', u'今天', u'吃饭', u'感觉', u'火锅', u'美食']
    type2 = [u'难受', u'感冒', u'今天', u'现在', u'痛苦', u'感觉']
    type3 = [u'旅行', u'风景', u'我们', u'颜色', u'博文', u'场景']
    type4 = [u'我们', u'学校', u'学会', u'同学', u'纪念', u'考试']


    sim_res=[]
    #利用训练好的词向量模型计算相似度
    for line in li:
        temp=[]
        sim1 = cal_n_similarity(type1, line, model)
        sim2 = cal_n_similarity(type2, line, model)
        sim3 = cal_n_similarity(type3, line, model)
        sim4 = cal_n_similarity(type4, line, model)
        # print('1:'+str(sim1)+' '+'2:'+str(sim2)+' '+'3:'+str(sim3)+' '+'4:'+str(sim4))
        temp.append(str(sim1))
        temp.append(str(sim2))
        temp.append(str(sim3))
        temp.append(str(sim4))
        sim_res.append(temp)

    return sim_res

def cal_n_similarity(w1,w2,model):
    _w1 = [w for w in w1 if w in model.vocab.keys()]
    _w2 = [w for w in w2 if w in model.vocab.keys()]
    sim = 0.1
    if len(_w2):
        sim = model.n_similarity(_w1, _w2)
    # print(_w1)
    # print(_w2)
    return sim


def writeback_sim(sim_res,labels):
    f=open('similiardata.txt','w',encoding='utf=8')
    l=len(sim_res)
    for i in range(l):
        f.write(str(labels[i])+'\t'+' '.join(sim_res[i])+'\n')

    f.close()

def read_svmd():
    f=open('similiardata.txt','r',encoding='utf-8')
    lines=f.readlines()
    line=[s.rstrip('\n') for s in lines]
    x=[]
    y=[]
    for s in line:
        sp=s.split('\t')
        y.append(int(sp[0]))
        x.append(list(map(float,list(sp[1].split(' ')))))

    return x,y

def svm_train(x,y):
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.3, random_state=0)

    # print(x_train)
    # print(x_test)
    # print(y_train)
    model =OneVsRestClassifier(svm.SVC(C=1.0,kernel='rbf',gamma='scale'))
    model.fit(x_train, y_train)

    # y_score = model.decision_function(x_test)

    y_pred =model.predict(x_test)

    confusion_matrix(y_test, y_pred)

    # accuracy_score(y_test, y_pred)
    # recall_score(y_test, y_pred, average='micro')
    # f1_score(y_test, y_pred, average='micro')

    print(classification_report(y_test, y_pred, digits=3))



    # for kernel in ['linear', 'rbf']:
    #     svr = SVR(kernel=kernel)
    #     svr.fit(x_train, y_train)
    #     # 打印拟合过程参数
    #     # print(svr.fit(x_train, y_train))
    #     # 打印训练集得分
    #
    #     wex=svr.predict(x_test)
    #     print(wex)


if __name__ == '__main__':
    # labels,contents=readtrain('eventdata1.txt')
    # li,labels=cutwords(contents,labels)
    #
    # sim_res=calsimilar(li)
    # writeback_sim(sim_res,labels)
    x,y=read_svmd()
    # print(np.array(x))
    # print(np.array(y))
    svm_train(x,y)

    # usemodel()