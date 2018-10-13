
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset From Drive
Data = pd.read_csv('DataSet\ElectionData2.csv',  sep=',')


# Finding No of Rows
len(Data)

# Create Data into Binary form(0,1)
Data = pd.get_dummies(Data, columns=['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
                               'immigration','synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
                               'crime', 'duty-free-exports', 'export-administration-act-south-africa'])

# Generate the no of Rows from data
Data = Data.sample(frac=1)

# Train & Test Data
train_data = Data[:300]
test_data = Data[300:]

# Drop & Save Predictable attribute (From Train Data)
data_train_att = train_data.drop(['Class-Name'], axis=1)
data_train_ClassName = train_data['Class-Name']

# Drop & Save Predictable attribute (From Test Data)
data_test_attr = test_data.drop(['Class-Name'], axis=1)
data_test_classname = test_data['Class-Name']

# Drop Predictable attribute into data_attr
data_attr = Data.drop(['Class-Name'],axis=1)

# Save Predictable attribute into data_attr
data_classname = Data['Class-Name']


decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
# Generate Tree using skLearn Library
decision_tree = decision_tree.fit(data_train_att, data_train_ClassName)

# Export a decision tree in DOT format.
graph_data = tree.export_graphviz(decision_tree, out_file=None, label="all", impurity=False, proportion=True,feature_names=list(data_train_att), class_names=["republican", "democrat"],filled=True, rounded=True)
graph = graphviz.Source(graph_data)
graph

# Export a decision tree in DOT format File.
tree.export_graphviz(decision_tree, out_file="house-votes.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(data_train_att), class_names=["republican", "democrat"],
                     filled=True, rounded=True)
# Accuracy Calculation
decision_tree.score(data_test_attr, data_test_classname)
accuracy = cross_val_score(decision_tree, data_attr, data_classname, cv=5)


print("Accuracy Of Tree : %0.3f (-/+ %0.3f)" % (accuracy.mean(), accuracy.std() * 2))

# initialize empty matrix of 25x3
depth_accuracy= np.empty((25,3), float)
i = 0

# Generate different number of trees and test on our system
# Save the accuracy
for max_depth in range(1, 25):
    decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    accuracy = cross_val_score(decision_tree, data_attr, data_classname, cv=5)
    depth_accuracy[i, 0] = max_depth
    depth_accuracy[i, 1] = accuracy.mean()
    depth_accuracy[i, 2] = accuracy.std() * 2
    i += 1
depth_accuracy

# Plotting
fig, ax = plt.subplots()
ax.errorbar(depth_accuracy[:,0], depth_accuracy[:,1], yerr=depth_accuracy[:,2])
plt.show()
