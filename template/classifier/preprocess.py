#-*- coding: utf-8 -*-
import csv
import numpy
import pandas

# for csv all file
my_dataframe = pandas.read_csv('../testdata/csv_test.csv')
print my_dataframe.columns
train_data = my_dataframe.values[:, :-1]
train_label = my_dataframe.values[:, -1]
print train_data
print train_label

# for csv each line
reader = csv.reader(open('../testdata/csv_test.csv', 'r'))
for row in reader:
    print row

# for csv all number
my_matrix = numpy.loadtxt(open('../testdata/csv_test.csv', "r"), delimiter=",", skiprows=1)
print my_matrix

# for pkl