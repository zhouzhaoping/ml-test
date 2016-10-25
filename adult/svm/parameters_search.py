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

from data_preprocess import row2dict
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame

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

#特征选择
vec = DictVectorizer()
train_x = vec.fit_transform(train_data[:10000]).toarray()
print 'select %d features complete!' % (len(train_x[0]))

#参数优化
start = time.clock()
model = SVC(kernel='rbf', probability=True)

param_grid = {'C': [10, 50, 80, 100], 'gamma': [0.001, 0.0005, 0.0003, 0.0001]}
if __name__ == '__main__':
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(train_x, train_y[:10000])

    #best_parameters = grid_search.best_estimator_.get_params()
    #for para, val in best_parameters.items():
    #    print para, val
    results = DataFrame(grid_search.cv_results_)
    results = results.sort(columns='rank_test_score').head(10)
    print results.loc[:, ['rank_test_score', 'params', 'mean_test_score']]
    print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

end = time.clock()
print "searching SVM parameters : %f s" % (end - start)