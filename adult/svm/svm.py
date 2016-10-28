# -*-coding:utf-8-*-
import csv
import time

from data_preprocess import row2dict
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing

#加载数据
reader = csv.reader(open('../adult.data', 'r'))
train_data = []
train_y = []
for row in reader:
    #去除训练集中的未知项
    #if ' ?' not in row:
    train_data.append(row2dict(row))
    train_y.append(0 if row[14] == ' <=50K' else 1)
print 'load %d train_data complete!' % (len(train_data))

#数据归一化
scaler = preprocessing.StandardScaler().fit(train_data)
train_x = scaler.transform(train_data)
print 'z-core complete!'

#算法执行
start = time.clock()
model = SVC(kernel='rbf', C=6000, gamma=0.3, probability=True, max_iter=-1)#参数优化详见parameters_search.py
model.fit(train_x, train_y)
end = time.clock()
print "running SVM: %f s" % (end - start)

#加载测试集
reader = csv.reader(open('../adult.test', 'r'))
test_data = []
test_y = []
for row in reader:
    test_data.append(row2dict(row))
    test_y.append(0 if row[14] == ' <=50K.' else 1)
print 'load %d test_data complete!' % (len(test_data))

#预测
train_y = scaler.transform(test_data)
predict = model.predict(train_y)
print(metrics.classification_report(test_y, predict))

#precision = metrics.precision_score(test_y, predict)
#recall = metrics.recall_score(test_y, predict)
#print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
#accuracy = metrics.accuracy_score(test_y, predict)
#print 'accuracy: %.2f%%' % (100 * accuracy)


