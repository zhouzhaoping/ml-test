# -*- coding: utf-8 -*-
import adaboost

# dataMat,classLabels=stumpTree.loadData()
# stumpTree.adaBoostTrainDS(dataMat,classLabels)

dataMat, classLabels = adaboost.file2Matrix('/home/lvsolo/python/adaBoosting/horseColicTraining2.txt')
weakClassify = adaboost.adaBoostTrainDS(dataMat, classLabels, 50)
dataTest, testLabels = adaboost.file2Matrix('/home/lvsolo/python/adaBoosting/horseColicTest2.txt')
adaboost.adaBoostTest(dataTest, testLabels, weakClassify)
