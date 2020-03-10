import numpy as np
import lightgbm as lgb
from sklearn import svm
from sklearn.metrics import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#读取需要训练的数据
def read_data(file):
    f=open(file,'r',encoding='utf-8')
    lines=f.readlines()
    line=[s.rstrip('\n') for s in lines]
    x=[]
    y=[]
    for s in line:
        sp=s.split('\t')
        y.append(int(sp[0]))
        x.append(list(map(float,list(sp[1].split(' ')))))

    return x,y

def lightgb_train(x,y):
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.3, random_state=0)

    train_data = lgb.Dataset(x_train, label=y_train)
    validation_data = lgb.Dataset(x_test, label=y_test)

    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 8,
        'objective': 'multiclass',
        'num_class': 5}
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred = clf.predict(x_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    print(y_pred)
    print(accuracy_score(y_test, y_pred))

    # clf = lgb.train(params, train_data, valid_sets=[validation_data])
    #
    # y_pred = clf.predict(x_test)

    #  3、经典-精确率、召回率、F1分数
    # precision_score(y_test, y_pred, average='micro')
    # recall_score(y_test, y_pred, average='micro')
    # f1_score(y_test, y_pred, average='micro')

    # 4、模型报告
    print(classification_report(y_test, y_pred))



if __name__ == '__main__':
    x,y=read_data('similiardata_new2.txt')
    lightgb_train(x,y)



