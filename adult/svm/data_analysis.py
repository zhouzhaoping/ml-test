# -*-coding:utf-8-*-
import csv
import operator

reader = csv.reader(open('../adult.data', 'r'))
name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
missing_sum = 0
missing_name = []
count_detail = {}
capital_gain_count = 0
capital_loss_count = 0
index = 13
incomesum = {}
personsum = {}
for i in range(len(name)):
    missing_name.append(0)

for row in reader:
    #计算评价工资
    if row[index] not in incomesum.keys():
        incomesum[row[index]] = 0 if row[14] == ' <=50K' else 1
        personsum[row[index]] = 1
    else:
        incomesum[row[index]] += 0 if row[14] == ' <=50K' else 1
        personsum[row[index]] += 1
    flag = 0
    for i in range(len(row)):
        #统计缺失项
        if row[i] == ' ?':
            missing_name[i] += 1
            flag = 1
    if flag == 0:
        #统计离散变量
        for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
            if row[i] in count_detail.keys():
                count_detail[row[i]] += 1
            else:
                count_detail[row[i]] = 1
        #统计投资
        if long(row[10]) != 0:
            capital_gain_count += 1
        if long(row[11]) != 0:
            capital_loss_count += 1
    else:
        missing_sum += 1

#计算哪个特征中丢失了多少信息
for i in range(len(name)):
    if missing_name[i] != 0:
        print "%s missing rate: %.2f%%" % (name[i], float(missing_name[i]) * 100 / reader.line_num)

#计算离散变量们的特征分布
for (k, v) in sorted(count_detail.iteritems(), key=operator.itemgetter(1), reverse=True):
    print "%s contain rate: %.2f%%" % (k, float(v) * 100 / (reader.line_num - missing_sum))

#计算capital不为零的百分比
print 'capital_gain_count: %.2f%%' % (capital_gain_count * 100 / float(reader.line_num - missing_sum))
print 'capital_loss_count: %.2f%%' % (capital_loss_count * 100 / float(reader.line_num - missing_sum))

#计算平均工资
for k in incomesum.keys():
    incomesum[k] = (float)(incomesum[k]) / personsum[k]
for (k, v) in sorted(incomesum.iteritems(), key=operator.itemgetter(1), reverse=True):
    print "%s's perincome : %.2f%% , sample : %d" % (k, v, personsum[k])

for (k, v) in sorted(incomesum.iteritems(), key=operator.itemgetter(1), reverse=True):
    print "%s," % k,
    #print "\"%s\"," % k,