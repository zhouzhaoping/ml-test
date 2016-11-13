[检测高收入人群](http://archive.ics.uci.edu/ml/datasets/Adult)  
####svm文件夹：  
svm分类的实现  

- data_analysis.py:  
数据分析，对数据进行统计，找出一些规律方便后面的数据处理和特征值选择
- feature_select.py:  
使用随机深林等算法计算数据中的相关度，为之后的特征选择做出一些建议
- data_preprocess.py:  
数据预处理，为特征分组、编号
- parameters_search.py:  
使用网格搜索和暴力搜索来对svm参数进行搜索，找到一个更好的参数
- svm.py:  
执行主要算法流程，用svm方式训练模型，并对测试集进行预测