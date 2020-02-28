import pandas as pd
import jieba
import jieba.posseg as psg
import re
from gensim.models import word2vec
import sys

# 预处理数据
def dealdata(li):
    numsten = [str(s).strip() for s in li]
    news = [k for k in numsten if k != '']
    nnews = [t.replace(' 我在:', '') for t in news]
    r1=[t.replace(' 我在这里:', '') for t in nnews]
    r2=[t.replace('下载地址:','') for t in r1]
    # r3=[t.replace('//:','') for t in r2]

    return r2

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

# 中文分词和词性
def cutwords(li):
    for se in li:
        print([(x.word,x.flag) for x in psg.lcut(se)])

def loadstopwords():
    stopwords=[]
    with open('stop_words.txt',encoding='utf-8-sig') as fr:
        for line in fr:
            stopwords.append(str(line.replace('\n','')))

    return stopwords

#进行中文分词和去除停顿词
def clearstop(li):
    jiebaseg=[]
    length=len(li)
    for i in range(0,length):
        temp=[' '.join(jieba.cut(li[i],cut_all=False))]
        st=temp[0].split(' ')
        jiebaseg.append(st)

    # print(jiebaseg)

    #去除停顿词
    stopwords=loadstopwords()
    result=[]

    for i in range(len(jiebaseg)):
        t=[]
        words=jiebaseg[i]
        for j in range(len(words)):
            if words[j] not in stopwords:
                t.append(words[j])
        result.append(t)

    f=open('words_res.txt','w',encoding='utf-8')
    for r in result:
        s = str(r).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        print(s)
        f.write(s)
    f.close()
    return result

def writeback(stoplist):
    f=open('eventdata.txt','w',encoding='utf-8')
    i=0
    for list in stoplist:
        for s in list:
            f.write(str(i)+'\t'+s + '\n')
        i+=1
    f.close()

if __name__ == '__main__':
    ty,words=read_ewords('eventwords.txt')
    data = pd.read_csv('w1.csv', engine='python')
    fr = data['content']
    fr=fr.dropna()

    stes=fr.tolist()
    newstr=dealdata(stes)
    length=len(ty)
    li = takesent(newstr,words,length)

    #所有类别的list集合
    totallist=[]

    totallist.append(list(set(li[0])))
    totallist.append(list(set(li[1])))
    totallist.append(list(set(li[2])))
    totallist.append(list(set(li[3])))
    #所有包含触发词的集合
    newtolist=[]
    for ll in totallist:
        newstoplist = []
        for st in ll:
            newstr = re.sub(r'(//[:：].*)', '', st)
            newstoplist.append(newstr)
        newtolist.append(newstoplist)
    relist=[]
    for l in newtolist:
        relist.append(list(filter(None, l)))
    # relist=list(filter(None,newstoplist))
    print(len(relist[3]))
    writeback(relist)
    # clearstop(stoplist)
    # print(len(list[0]))
    # cutwords(list[0])


