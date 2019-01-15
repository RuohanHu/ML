import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

'''
old_file = open("pa2train.txt")
new_file = open("new_pa2train.txt", "w")

for line in old_file:
    d = line.split()
    for i in range(0, 23):
        if i != 21:
            d[i] = d[i] + ", "
    new_file.write(
        d[22] + d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8] + d[9] + d[10] + d[11] + d[12] + d[13] + d[14] + d[15] + d[16] + d[17] + d[18] + d[19] + d[20] + d[21] + 
        "\n")
'''

training_data = pd.read_csv("new_pa2train.csv", sep=',', header=None)

print(len(training_data))
print(training_data.shape)

X = training_data.values[:,1:23]
Y = training_data.values[:,0]


clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(X, Y)

tree.export_graphviz(clf_entropy, out_file='tree.dot')
