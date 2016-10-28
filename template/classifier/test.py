#-*- coding: utf-8 -*-

import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
from classifiers import *

def read_data(data_file):
    f = open(data_file, "rb")
    train, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    data_file = "data_after_reprocess"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier
    }

    print 'reading training and testing data...'
    train_x, train_y, test_x, test_y = read_data(data_file)
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)

    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
        accuracy = metrics.accuracy_score(test_y, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))