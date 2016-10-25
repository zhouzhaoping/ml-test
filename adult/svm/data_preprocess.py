# -*-coding:utf-8-*-
import csv

workclass_value = [" Private", " Self-emp-not-inc", " Self-emp-inc", " Federal-gov", " Local-gov", " State-gov", " Without-pay", " Never-worked"]
marital_status_value = [" Married-civ-spouse", " Divorced", " Never-married", " Separated", " Widowed", " Married-spouse-absent", " Married-AF-spouse"]
occupation_value = [" Tech-support", " Craft-repair", " Other-service", " Sales", " Exec-managerial", " Prof-specialty", " Handlers-cleaners", " Machine-op-inspct", " Adm-clerical", " Farming-fishing", " Transport-moving", " Priv-house-serv", " Protective-serv", " Armed-Forces"]
relationship_value = [" Wife", " Own-child", " Husband", " Not-in-family", " Other-relative", " Unmarried"]
race_value = [" White", " Asian-Pac-Islander", " Amer-Indian-Eskimo", " Other", " Black"]
native_country_value = [" United-States", " Cambodia", " England", " Puerto-Rico", " Canada", " Germany", " Outlying-US(Guam-USVI-etc)", " India", " Japan", " Greece", " South", " China", " Cuba", " Iran", " Honduras", " Philippines", " Italy", " Poland", " Jamaica", " Vietnam", " Mexico", " Portugal", " Ireland", " France", " Dominican-Republic", " Laos", " Ecuador", " Taiwan", " Haiti", " Columbia", " Hungary", " Guatemala", " Nicaragua", " Scotland", " Thailand", " Yugoslavia", " El-Salvador", " Trinadad&Tobago", " Peru", " Hong", " Holand-Netherlands"]

def string2code(value):
    dict = {}
    for i in range(len(value)):
        dict[value[i]] = i
    return dict

workclass_map = string2code(workclass_value)
marital_status_map = string2code(marital_status_value)
occupation_map = string2code(occupation_value)
relationship_map = string2code(relationship_value)
race_map = string2code(race_value)
native_country_map = string2code(native_country_value)

#缺失数据补齐
workclass_map[' ?'] = workclass_map[" Private"]
occupation_map[' ?'] = len(occupation_map)
native_country_map[' ?'] = native_country_map[" United-States"]

def row2dict(row):
    dict = {}
    dict['age'] = long(row[0])
    dict['workclass'] = workclass_map[row[1]]
    #抽样权重不要dict['fnlwgt'] = long(row[2])
    #与education-num重复dict['education'] = row[3]
    dict['education-num'] = long(row[4])
    dict['marcital-status'] = marital_status_map[row[5]]
    dict['occupation'] = occupation_map[row[6]]
    dict['relationship'] = relationship_map[row[7]]
    dict['race'] = race_map[row[8]]
    dict['sex'] = 0 if row[9] == ' Male' else 1
    dict['capital-gain'] = long(row[10])
    dict['capital-loss'] = long(row[11])
    dict['hours-per-week'] = long(row[12])
    dict['native-country'] = native_country_map[row[13]]
    #dict['income'] = 0 if row[14] == ' <=50K' else 1
    return dict
