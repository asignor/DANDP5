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
sys.path.append("../tools/")
import numpy as np
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
# Please see feature selection narrative in Addendum I

features_list = ['poi',
'salary',
#'my_feature' used to be here, but ended up not used
'deferral_payments',
'bonus',
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



### Remove outliers 
#this points are not people
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### Store to my_dataset for easy export below.
my_dataset = data_dict

#creation of my_feature - not utilized
for name in my_dataset:
    try:    
        my_dataset[name]['my_feature'] = my_dataset[name]['exercised_stock_options'] /          my_dataset[name]['total_stock_value']
    except: my_dataset[name]['my_feature'] = 0

from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, remove_NaN=True, 
sort_keys = True, remove_all_zeroes=True,)
labels, features = targetFeatureSplit(data)

# Addressing sparsity
# remove points with 0 salary
clean_data = []
i = 0
for datapoint in data:
    if (datapoint[1] > 0):
        clean_data.append(datapoint)
data = clean_data

# transform negative feature 'deferred_income'
clean_data=[]
for datapoint in data:
    clean_datapoint = []
    for i in range(3):
        clean_datapoint.append(datapoint[i])
    clean_datapoint.append((-1)*datapoint[4])
    for i in range(5,11):
        clean_datapoint.append(datapoint[i])

print len(data), "datapoints remaining"
print "number features utilized:", len(features_list)

# balance of classes after data treatment
bal = [0,0] #[non_poi, poi]
for datapoint in data:
    if datapoint[0] == 1:
        bal[1] += 1
    else: bal[0] += 1
print bal

## Split train/test
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# normalizing data 

features= np.array(features)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

##### Task 3: Create new feature(s)

from sklearn.decomposition import PCA
nPCs = 7
pca = PCA(n_components=nPCs)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

###### Task 4: Try a variety of classifiers
# see report, picked RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

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
 
params = {
#'class_weight': ['None', 'balanced_subsample', 'balanced']
#'criterion': ['entropy', 'gini']
#'max_depth': [10, 20, 40,45,50]
#'max_features': [4, 5, 6,7,None]
#'max_leaf_nodes': [40, 43, 45, 50]
#'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
#'n_estimators': [8, 9, 10]
#'n_jobs': [1, 2, 4, 8, 16]
}

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
rfc = RandomForestClassifier(
    criterion = 'entropy', 
    class_weight = 'balanced',
    max_depth = 40, 
    max_features = 7, 
    min_samples_split = 2,
    min_weight_fraction_leaf=0.1, 
    n_estimators = 8)
cross_val = StratifiedShuffleSplit(labels_train, 3, random_state = 42)
clf = GridSearchCV(rfc, params, scoring="f1", cv=cross_val)
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

print 'best classifier:'
print clf.best_estimator_
 
dump_classifier_and_data(clf, my_dataset, features_list)

#the beeps are to alert when the script is done running, 
#because it takes so long
import winsound
winsound.Beep(1200,1500)
winsound.Beep(200,200)
winsound.Beep(1200,1500)