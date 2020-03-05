import pandas as pd
import re



#写回文本
def writeback(li,file):
    f = open(file, 'w', encoding='utf-8')
    for s in li:
        f.write(s+'\n')
    f.close()


#去除一些无关的字符
def washdata(weibodata,filename):
    ls=[str(s) for s in weibodata]
    ls1=[re.sub(r'(//[:：].*)', '', s) for s in ls]
    ls2=[re.sub(r'(我在[:：].*)', '', s) for s in ls1]
    ls3=[re.sub(r'我在这里[:：].*','',s) for s in ls2]
    ls4=[re.sub(r'下载地址[:：].*','',s) for s in ls3]
    ls5=[re.sub(r'&gt.*','',s) for s in ls4]

    ls6=[s.strip() for s in ls5]

    #去除空字符串
    rels=list(filter(None,ls6))

    writeback(rels,filename)
    return rels

    # for s in weibodata:
    #     newstr = re.sub(r'(//[:：].*)', '', s)
    #     newdata.append(newstr)
    #
    # for s in weibodata:
    #     newstr=s.strip()
    #     tempdata.append(newstr)


if __name__ == '__main__':
    weibo_data1 = pd.read_csv('w1.csv', encoding='utf-8')
    weibo_data2 = pd.read_csv('w2.csv', encoding='utf-8')
    #获取初始数据并清除空元素
    fr1 = weibo_data1['content']
    fr2 = weibo_data2['content']
    fr1 = fr1.dropna()
    fr2 = fr2.dropna()

    # print(fr1.isnull)
    # print(fr2.isnull)

    stes1=fr1.tolist()
    stes2=fr2.tolist()

    print(len(stes1))
    washdata(stes1,'filtdata1.txt')
    washdata(stes2,'filtdata2.txt')



