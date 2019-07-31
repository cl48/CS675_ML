#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import array
import copy
import random
from sklearn import svm

seed = 100
samp = 0.4
gamma = 0.1
iter = 10

def data_set(feature, data):
    newData = [[row[feature[0]]] for row in data]

    feature.remove(feature[0])
    length = len(feature)
    for _ in range(0, length, 1):
        data_set = [[row[feature[0]]] for row in data]
        newData = [x + y for x, y in zip(newData, data_set)]
        feature.remove(feature[0])
    return newData

def data_split(data, labels, test_samp=samp):
    random.seed(seed)
    num_of_test_data = len(data) * test_samp
    indicies = random.sample(range(len(data)), int(num_of_test_data))
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for feat_i in range(len(data)):
        if feat_i not in indicies:
            x_train.append(data[feat_i])
            y_train.append(labels[feat_i])
        else:
            x_test.append(data[feat_i])
            y_test.append(labels[feat_i])
    return x_train, x_test, y_train, y_test


def pearson_correlation(x, y, fi):
    totalx = 0
    totalx2 = 0
    ro = len(x)
    co = len(x[0])
    switch = 0
    pc = array.array("f")
    for i in range(0, co, 1):
        switch += 1
        totaly = 0
        totaly2 = 0
        totalxY = 0
        for j in range(0, ro, 1):
            if (switch == 1):
                totalx += y[j]
                totalx2 += y[j] ** 2
            totaly += x[j][i]
            totaly2 += x[j][i] ** 2
            totalxY += y[j] * x[j][i]
        r = (ro * totalxY - totalx * totaly) / ((ro * totalx2 - (totalx ** 2)) * (ro * totaly2 - (totaly ** 2))) ** (0.5)
        pc.append(abs(r))

    store_print = array.array("f")
    feature_store = array.array("i")
    for i in range(0, fi, 1):
        selectfeatures = max(pc)
        store_print.append(selectfeatures)
        featureIndex = pc.index(selectfeatures)
        pc[featureIndex] = -1
        feature_store.append(featureIndex)
    return feature_store


# Read data file
data_file = "C:/Users/Visitor/CS 675/traindata"
data = []
with open(data_file, "r") as file:
    for line in file:
        s = line.split()
        l = array.array("i")
        for i in s:
            l.append(int(i))
        data.append(l)

# Read labels from file
labels = "C:/Users/Visitor/CS 675/test labels.txt"
trainlabels = array.array("i")
with open(labels, "r") as file:
    for line in file:
        s = line.split()
        trainlabels.append(int(s[0]))

feature_count = 8
rows = len(data)
cols = len(data[0])
rows = len(trainlabels)

# Dimensionality Reduction
pc_features = pearson_correlation(data, trainlabels, 2000)
saved_features = copy.deepcopy(pc_features)
updated_data = data_set(pc_features, data)

# SVC Model
svc = svm.SVC(gamma=gamma)
accur_array = array.array("f")
feature_array = []
accuracy_svm = 0

print("Running SVC algorithm and accuracy:")

for i in range(iter):
    print("\nIteration # ", i + 1)
    x_train, x_test, y_train, y_test = data_split(
        updated_data, trainlabels, test_samp=0.3)
    my_correl = pearson_correlation(x_train, y_train, feature_count)
    feature_array.append(my_correl)
    argument = copy.deepcopy(my_correl)
    data_fea = data_set(argument, x_train)
    svc.fit(data_fea, y_train)
    features = pearson_correlation(x_test, y_test, feature_count)
    test_features = data_set(features, x_test)
    len_test_features = len(test_features)
    counter_svm = 0
    my_counter = 0
    for j in range(0, len_test_features, 1):
        svc_pred_labels = int(svc.predict([test_features[j]]))
        if (svc_pred_labels >= 3):
            final_predicted_labels = 1
        elif (svc_pred_labels <= 1):
            final_predicted_labels = 0
        else:
            final_predicted_labels = svc_pred_labels
        if (svc_pred_labels == y_test[j]):
            counter_svm += 1
    accuracy_svm += counter_svm / len_test_features
    accur_array.append(my_counter / len_test_features)
topaccuracy = max(accur_array)
bestInd = accur_array.index(topaccuracy)
bestFeatures = feature_array[bestInd]
print("\nNumber of Features: ", feature_count)
all_features = array.array("i")
for i in range(0, feature_count, 1):
    realIndex = saved_features[bestFeatures[i]]
    all_features.append(realIndex)
print("\nThe features are: ", all_features)

# Accuracy check
a_feature = copy.deepcopy(all_features)
accuracies = data_set(a_feature, data)
svc.fit(accuracies, trainlabels)
c = 0
k = len(accuracies)
for i in range(0, k, 1):
    svc_pred_labels = int(svc.predict([accuracies[i]]))
    if (svc_pred_labels >= 3):
        final_predicted_labels = 1
    if (svc_pred_labels <= 1):
        final_predicted_labels = 0
    else:
        final_predicted_labels = svc_pred_labels
    if (svc_pred_labels == trainlabels[i]):
        c += 1
accur = c / k
print("\nAccuracy: ", accur * 100)

# Read Test
testfile = "C:/Users/Visitor/CS 675/testdata"
testdata = []
with open(testfile, "r") as file:
    for line in file:
        s = line.split()
        l = array.array("i")
        for i in s:
            l.append(int(i))
        testdata.append(l)
deep_copy = copy.deepcopy(all_features)
updated_test_data = data_set(deep_copy, testdata)

# create a file
predlbl = open("C:/Users/Visitor/CS 675/predlbls", "w+")
for i in range(0, len(updated_test_data), 1):
    a_labl = int(svc.predict([updated_test_data[i]]))
    predlbl.write(str(a_labl) + " " + str(i) + "\n")
    
# create feature file
with open('C:/Users/Visitor/CS 675/feature', 'w+') as f:
    for item in all_features:
        f.write("%s\n" % item)


# In[ ]:




