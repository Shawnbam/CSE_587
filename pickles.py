#!/usr/bin/env python
# coding: utf-8
import io
# In[ ]:


import pickle

# randomForest = None
# with open('randomForest.pkl', 'wb') as f:
#     randomForest = pickle.load(f)
try:
    with open('randomForest.pkl', 'rb') as f:
        randomForest = pickle.load(f)
except io.UnsupportedOperation as e:
    print(f"Error: {e}")

def getRandomForestClassification(X_Test_New):
    prediction = randomForest.predict(X_Test_New)
    return prediction

logmodel = None
with open('logmodel.pkl', 'rb') as f:
    logmodel = pickle.load(f)
def getLogisticClassification(X_Test_New):
    prediction = logmodel.predict(X_Test_New)
    return prediction

clf = None
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
def getCLFClassification(X_Test_New):
    prediction = clf.predict(X_Test_New)
    return prediction

NB = None
with open('NB.pkl', 'rb') as f:
    NB = pickle.load(f)
def getNBClassification(X_Test_New):
    prediction = NB.predict(X_Test_New)
    return prediction

gdboost = None
with open('gdboost.pkl', 'rb') as f:
    gdboost = pickle.load(f)
def getGDBoostClassification(X_Test_New):
    prediction = gdboost.predict(X_Test_New)
    return prediction

tree = None
with open('tree.pkl', 'rb') as f:
    tree = pickle.load(f)
def getTreeClassification(X_Test_New):
    prediction = tree.predict(X_Test_New)
    return prediction

