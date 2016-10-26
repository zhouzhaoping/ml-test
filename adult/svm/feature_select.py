# -*-coding:utf-8-*-
import csv

import numpy as np
from data_preprocess import row2dict
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

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

#特征选择(随机深林计算相关度）
name = ["age", "workclass", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
X = np.array(train_data)
Y = np.array(train_y)
names = np.array(name)
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2", cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print sorted(scores, reverse=True)

#使用树算法
model = ExtraTreesClassifier()
model.fit(train_data, train_y)
scores = []
for i in range(len(names)):
     scores.append((round(model.feature_importances_[i], 3), names[i]))
print sorted(scores, reverse=True)

