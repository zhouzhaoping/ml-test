# -*-coding:utf-8-*-
'''
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm结束标准的精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
（11）verbose: 跟多线程有关，不大明白啥意思具体；
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
 ps：7,8,9一般不考虑。
'''
import csv
import time
import numpy as np
from data_preprocess import row2dict
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pandas import DataFrame
from scipy.stats import randint, expon
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def draw_report(results):
    x = []
    y = []
    z = []
    for i in range(1, len(results) + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            z.append(results['mean_test_score'][candidate])
            x.append(results['params'][candidate]['C'])
            y.append(results['params'][candidate]['gamma'])
    print 'C : ', x
    print 'gamma : ', y
    print 'score : ', z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:5], y[:5], z[:5], c='r') #绘制数据点
    ax.scatter(x[5:], y[5:], z[5:], c='y') #绘制数据点
    ax.set_zlabel('score') #坐标轴
    ax.set_ylabel('gamma')
    ax.set_xlabel('C')
    plt.show()

#加载数据
reader = csv.reader(open('../adult.data', 'r'))
train_data = []
train_y = []
for row in reader:
    train_data.append(row2dict(row))
    train_y.append(0 if row[14] == ' <=50K' else 1)
print 'load %d train_data complete!' % (len(train_data))

#数据归一化
scaler = preprocessing.StandardScaler().fit(train_data)
train_x = scaler.transform(train_data)

#参数优化（网格搜索or随机搜索）
start = time.clock()
model = SVC(kernel='rbf', probability=True)

#添加抽样过程
size = len(train_data) / 10;
X = np.column_stack((train_x, train_y))
random.shuffle(X)
X = X[:size]
train_x = [x[:-1] for x in X]
train_y = [x[-1] for x in X]

if __name__ == '__main__':
    n_iter_search = 50
     # param_dist = {'kernel': ['poly', 'rbf', 'sigmoid'], 'C': expon(scale=100),
     #              'gamma': expon(scale=.1), 'class_weight': [None, 'balanced']}
    # param_dist = {'kernel': ['rbf'], 'C': expon(scale=100),
    #               'gamma': expon(scale=.1), 'class_weight': [None]}
    param_dist = {'kernel': ['rbf'], 'C': range(1000, 10000),
                  'gamma': [float(x) / 10 for x in range(1, 50)], 'class_weight': [None]}
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)
    random_search.fit(train_x, train_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.clock() - start), n_iter_search))
    report(random_search.cv_results_, n_top=10)
    draw_report(random_search.cv_results_)

# param_grid = {'C': range(1, 10000), 'gamma': [float(x) / 100 for x in range(1, 1000)]}
# if __name__ == '__main__':
#     grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
#     grid_search.fit(train_x, train_y)
#
#     #best_parameters = grid_search.best_estimator_.get_params()
#     #for para, val in best_parameters.items():
#     #    print para, val
#     results = DataFrame(grid_search.cv_results_)
#     results = results.sort(columns='rank_test_score').head(10)
#     print results.loc[:, ['rank_test_score', 'params', 'mean_test_score']]
#     print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
#     draw_report(grid_search.cv_results_)

end = time.clock()
print "searching SVM parameters : %f s" % (end - start)