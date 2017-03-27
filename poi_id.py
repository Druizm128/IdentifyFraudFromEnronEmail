#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list = ['poi','total_payments','total_stock_value','fraction_to_poi_messages','fraction_from_poi_messages']
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other','expenses','director_fees','exercised_stock_options','restricted_stock','restricted_stock_deferred','fraction_to_poi_messages','fraction_from_poi_messages']
# features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other','expenses','director_fees','total_payments','exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value','to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi','shared_receipt_with_poi','fraction_to_poi_messages','fraction_from_poi_messages']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Removing total observations
data_dict.pop('TOTAL',0)

# Not POI outliers
data_dict.pop('BAXTER JOHN C',0)
data_dict.pop('BHATNAGAR SANJAY',0)
data_dict.pop('DIMICHELE RICHARD G',0)
data_dict.pop('FREVERT MARK A',0)
data_dict.pop('HORTON STANLEY C',0)
data_dict.pop('LAVORATO JOHN J',0)
data_dict.pop('MARTIN AMANDA K',0)
data_dict.pop('PAI LOU L',0)
data_dict.pop('WHALEY DAVID A',0)
data_dict.pop('WHITE JR THOMAS E',0)

# POI outliers
#data_dict.pop('HIRKO JOSEPH',0)
data_dict.pop('LAY KENNETH L',0)
#data_dict.pop('RICE KENNETH D',0)
data_dict.pop('SKILLING JEFFREY K',0)
#data_dict.pop('YEAGER F SCOTT',0)


### Task 3: Create new feature(s)
# Fraction of emails written to POI and received from POI from total emails sent and received respectively

for name in data_dict:
	total_messages = data_dict[name]['to_messages']
	to_poi = data_dict[name]['from_this_person_to_poi']

	if total_messages == 'NaN' or to_poi == 'NaN':
		data_dict[name]['fraction_to_poi_messages'] = 0
	else:
		data_dict[name]['fraction_to_poi_messages'] = float(to_poi) / float(total_messages)

	total_received = data_dict[name]['from_messages']
	from_poi = data_dict[name]['from_poi_to_this_person']

	if total_received == 'NaN' or from_poi == 'NaN':
		data_dict[name]['fraction_from_poi_messages'] = 0
	else:
		data_dict[name]['fraction_from_poi_messages'] = float(from_poi) / float(total_received)
 
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Data exploration
print
print "--------- Data Exploration ---------"
print 'obs. :', len(labels)
print 'POIs: ', np.array(labels).sum()
print 'non-POIs: ', len(labels) - np.array(labels).sum()
print 'features: ', len(features_list)-1
print features_list[1:len(features)]
print
print 'Descriptive satistics'

features_dict = {}


for i in range(1,len(features_list)):
	feature_name = features_list[i]
	feature_data = []
	for person in features:
		feature_data.append(person[i-1])
	features_dict[feature_name] = feature_data

datos = pd.DataFrame(features_dict)

for var in features_list[1:]:
	datos[var].hist()
	plt.title(var + ' historgram')
	plt.show()
	plt.boxplot(datos[var].values)
	plt.title(var + ' boxplot')
	plt.show()

print datos.describe()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def modelEvaluation(prediction, real_observations):
	print 'Classifier accuracy with testing set:'
	accuracyScore = accuracy_score(prediction, real_observations)
	print 'Accuracy: ', accuracyScore
	precisionScore = precision_score(prediction, real_observations)
	print 'Precision score: ', precisionScore
	recallScore = recall_score(prediction, real_observations)
	print 'Recall score: ', recallScore	

# Provided to give you a starting point. Try a variety of classifiers.
clf = GaussianNB()
#clf_NB = GaussianNB()
clf_SVC = SVC()
clf_tree = tree.DecisionTreeClassifier(min_samples_split=50)
clf_KVC = KNeighborsClassifier()
clf_logit = LogisticRegression()
'''
clf_NB.fit(features,labels)
pred_NB = clf_NB.predict(features)
print '\nGaussian Naive Bavyes'
modelEvaluation(pred_NB, labels)


clf_SVC.fit(features,labels)
pred_SVC = clf_SVC.predict(features)
print '\nSVM'
modelEvaluation(pred_SVC, labels)

clf_tree.fit(features,labels)
pred_tree = clf_tree.predict(features)
print '\nDescision Tree'
modelEvaluation(pred_tree, labels)

clf_KVC.fit(features,labels)
pred_KVC = clf_KVC.predict(features)
print '\nK-nearest neighbors'
modelEvaluation(pred_KVC, labels)


clf_logit.fit(features,labels)
pred_logit = clf_logit.predict(features)
print '\nLogit model'
modelEvaluation(pred_logit, labels)
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.5, random_state=42)
'''


print "------------ Fine tuning------------"


from sklearn.model_selection import KFold
kf = KFold(10, shuffle=False)
print kf

accuracy_score_list = []
precision_score_list = []
recall_score_list = []

print '\nGaussian Naive Bayes'
print "................................"
for train_index, test_index  in  kf.split(labels):
	features_train = [features[ii] for ii in train_index] 
	features_test  = [features[ii] for ii in test_index]
	labels_train   = [labels[ii] for ii in train_index]
	labels_test    = [labels[ii] for ii in test_index]

	
	clf.fit(features_train,labels_train)
	pred = clf.predict(features_test)
	
	#modelEvaluation(pred_NB, labels_test)
	accuracy_score_list.append(accuracy_score(pred, labels_test))
	precision_score_list.append(precision_score(pred, labels_test))
	recall_score_list.append(recall_score(pred, labels_test))
print 'Average results for a KFold 10 splits:'
print 'Accuracy score mean: ', np.array(accuracy_score_list).mean()
print 'Precision score mean: ',np.array(precision_score_list).mean()
print 'Recall score mean: ', np.array(recall_score_list).mean()
'''
	for train_index, test_index  in  kf.split(labels):
	features_train = [features[ii] for ii in train_index] 
	features_test  = [features[ii] for ii in test_index]
	labels_train   = [labels[ii] for ii in train_index]
	labels_test    = [labels[ii] for ii in test_index]

	#print "................................"
	clf_NB.fit(features_train,labels_train)
	pred_NB = clf_NB.predict(features_test)
	print '\nGaussian Naive Bayes'
	#modelEvaluation(pred_NB, labels_test)
	accuracy_score_list.append(accuracy_score(pred_NB, labels_test))
	precision_score_list.append(precision_score(pred_NB, labels_test))
	recall_score_list.append(recall_score(pred_NB, labels_test))
print
print 'Accuracy score mean: ', np.array(accuracy_score_list).mean()
print 'Precision score mean: ',np.array(precision_score_list).mean()
print 'Recall score mean: ', np.array(recall_score_list).mean()

	clf_SVC.fit(features_train,labels_train)
	pred_SVC = clf_SVC.predict(features_test)
	print '\nSupport Vector Machine'
	modelEvaluation(pred_SVC, labels_test)

	clf_tree.fit(features_train,labels_train)
	pred_tree = clf_tree.predict(features_test)
	print '\nDescision Tree'
	modelEvaluation(pred_tree, labels_test)

	clf_KVC = KNeighborsClassifier()
	clf_KVC.fit(features_train, labels_train)
	pred_KVC = clf_KVC.predict(features_test)
	print '\nK-nearest neighbors'
	modelEvaluation(pred_KVC, labels_test)

	clf_logit.fit(features_train, labels_train)
	pred_logit = clf_logit.predict(features_test)
	print '\nLogit model'
	modelEvaluation(pred_logit, labels_test)


'''


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)