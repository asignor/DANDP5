#!/usr/bin/python

"""
This code is part of a project in the Udacity Data Analyst Nanodegree
DANDP5 - Identifying fraud from Enron
Anna Signor
Created on Sun Mar 5 2017

@author: Anna
"""


from sklearn.preprocessing import MinMaxScaler
import sys
import pickle
import pandas as pd
sys.path.append("../tools/")
import numpy as np
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [
'poi',
'salary', 
'deferral_payments', 
'total_payments', 
'loan_advances', 
'bonus', 
'restricted_stock_deferred', 
'deferred_income', 
'total_stock_value', 
'expenses', 
'exercised_stock_options', 
'other', 
'long_term_incentive', 
'restricted_stock', 
'director_fees']

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, remove_NaN=True, 
sort_keys = True, remove_all_zeroes=True,)
labels, features = targetFeatureSplit(data)

#### Task 2: Remove outliers
## find any datapoints containing negative values
mydf = pd.DataFrame(data)
neg_list = []
for i in range(0,144):
    for j in range(0,14): #slicing here because 6 and 7 will be all negatives by nature of feature
        if mydf.loc[i,j] < 0:
            neg_list.append([i,j,mydf.loc[i,j]])
    for j in range(8,14):
        if mydf.loc[i,j] < 0:
            neg_list.append([i,j,mydf.loc[i,j]])
print neg_list
print len(neg_list)

# features 6 and 7 make sense as negatives, but we will have to eliminate datapoints 8 and 11
del features[8]
del features[11]
del labels [8]
del labels[11]

## Split train/test

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### normalizing data ###################################

features= np.array(features)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

##### Task 3: Create new feature(s)

from sklearn.decomposition import PCA
nPCs = 8
pca = PCA(n_components=nPCs)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

###### Task 4: Try a variety of classifiers
# see report, picked RandomForestClassifier


from sklearn.ensemble import RandomForestClassifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

####################RFC default parameters############################
#'bootstrap': True,
# 'class_weight': None,
# 'criterion': 'gini',
# 'max_depth': None,
# 'max_features': 'auto',
# 'max_leaf_nodes': None,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0,
# 'n_estimators': 10,
# 'n_jobs': 1,
# 'oob_score': False,
# 'random_state': None,
# 'verbose': 0,
# 'warm_start': False
#######################################################################

#Because of the long time it takes to run even a simple random forest 
#grid search, it is done in iterations.  
#One parameter and 3 values at a time, GridSearchCV is ran and 
#once the best value for a given parameter is determined,
#it is extracted via best_estimator and hard-coded into the 
#instance of RandomForestClassifier passed as the starter
#classifier to GridSearchCV.
#The dictionary below, that is commented out is the full grid
#used over the iterations. A "dummy" dictionary was then written
#in with the sole purpose of being able to display the code with 
#the validation. The same results should be achieved with
# `clf = RandomForestClassifier(bootstrap=True, 
#class_weight='balanced_subsample', criterion='entropy', 
#max_depth = 40, max_features = 7, max_leaf_nodes = 45, #min_samples_leaf = 5,
#min_samples_split = 2)` instead of the grid search.
 
#params = {
#'class_weight': ['None', 'balanced_subsample', 'balanced']
#'criterion': ['entropy', 'gini']
#'max_depth': [10, 20, 40,45,50]
#'max_features': [4, 5, 6,7,None]
#'max_leaf_nodes': [40, 43, 45, 50, 55, 60]
#'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
#'n_estimators': [8, 9, 10]
#'n_jobs': [1, 2, 4, 8, 16]
#}

params = {}

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
rfc = RandomForestClassifier(
bootstrap=True, 
class_weight='balanced_subsample',
criterion='entropy', 
max_depth = 40, 
max_features = 7, 
max_leaf_nodes = 45, 
min_samples_leaf = 5)
cross_val = StratifiedShuffleSplit(labels_train, 3, random_state = 42)

clf = GridSearchCV(rfc, params, scoring="f1", cv=cross_val)
clf.fit(features_train, labels_train)

clf1 = RandomForestClassifier()
#for comparison

import winsound
winsound.Beep(1200,500) #the beeps are here so I can multitask while tuning

test_classifier(clf, my_dataset, features_list)
test_classifier(clf1, my_dataset, features_list)

print 'best classifier:'
print clf.best_estimator_
 
dump_classifier_and_data(clf, my_dataset, features_list)
dump_classifier_and_data(clf1, my_dataset, features_list)

winsound.Beep(1200,1500)
winsound.Beep(200,200)
winsound.Beep(1200,1500)
