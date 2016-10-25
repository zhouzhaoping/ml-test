# -*-coding:utf-8-*-
import csv
import time
from data_preprocess import row2dict
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn import preprocessing

#加载数据
reader = csv.reader(open('../adult.data', 'r'))
train_data = []
train_y = []
for row in reader:
    #去除训练集中的未知项
    if ' ?' not in row:
        train_data.append(row2dict(row))
        train_y.append(0 if row[14] == ' <=50K' else 1)
print 'load %d train_data complete!' % (len(train_data))

#数据归一化
#print train_data.toarray()
#scaler = preprocessing.StandardScaler().fit(train_data)
#scaler.transform(train_data)

#特征选择
vec = DictVectorizer()
train_x = vec.fit_transform(train_data).toarray()
print 'select %d features complete!' % (len(train_x[0]))

#算法执行
start = time.clock()
model = SVC(kernel='rbf', C=50, gamma=0.0005, probability=True)#参数优化详见parameters_search.py
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
test_x = vec.transform(test_data).toarray()
print 'load %d test_data complete!' % (len(test_x))

#预测
predict = model.predict(test_x)
print(metrics.classification_report(test_y, predict))

#precision = metrics.precision_score(test_y, predict)
#recall = metrics.recall_score(test_y, predict)
#print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
#accuracy = metrics.accuracy_score(test_y, predict)
#print 'accuracy: %.2f%%' % (100 * accuracy)


