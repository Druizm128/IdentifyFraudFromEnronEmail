#!/usr/bin/python

#!/usr/bin/python
import sys
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import pprint


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other','expenses','director_fees','exercised_stock_options','restricted_stock','restricted_stock_deferred','fraction_to_poi_messages','fraction_from_poi_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Removing non person observations
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Task 3: Create new feature(s)
# Fraction of emails written to POI and received from POI from total emails sent and received respectively

for name in data_dict:
    from_poi_to_this_person = data_dict[name]['from_poi_to_this_person']
    to_messages = data_dict[name]['to_messages']
    
    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN':
        data_dict[name]['fraction_to_poi_messages'] = 0
    else:
        data_dict[name]['fraction_to_poi_messages'] = float(from_poi_to_this_person) / float(to_messages)

    from_this_person_to_poi = data_dict[name]['from_this_person_to_poi']
    from_messages = data_dict[name]['from_messages']

    if from_this_person_to_poi == 'NaN' or from_messages == 'NaN':
        data_dict[name]['fraction_from_poi_messages'] = 0
    else:
        data_dict[name]['fraction_from_poi_messages'] = float(from_this_person_to_poi) / float(from_messages)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Data exploration
print '\nExploratory data analysis and descriptive statistics:\n'
df = pd.DataFrame(data)
df.columns = features_list
print 'Number of Rows and Columns: ', df.shape
print '\nPOIs: ', df.poi.sum()
print 'non-POIs: ', df['poi'].size - df.poi.sum()
print '\nfeatures: ', len(features_list)-1
print '\nfeatures list: ', features_list[1:len(features)]

# POI barplot

sns.countplot(df.poi)
plt.title("Persons of interest\n(0 = No, 1 = Yes)")
plt.show()


# scatter plot matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Defines a function that returns a dictionary containing the results of comparing the predicted vs the observed labels.
def modelEvaluation(prediction, real_observations):
    accuracyScore = accuracy_score(prediction, real_observations)
    precisionScore = precision_score(prediction, real_observations)
    recallScore = recall_score(prediction, real_observations)
    f1Score = f1_score(prediction, real_observations)
    confusion = confusion_matrix(prediction, real_observations)
    
    results = {'Accuracy_score': accuracyScore,
               'Precision_score': precisionScore, 
               'Recall_score': recallScore, 
               'F_score': f1Score,
               'Predictions':len(labels_test),
               'True positives': confusion[1,1],
               'True negatives': confusion[0,0],
               'False positives': confusion[0,1],
               'False negatives': confusion[1,0]
              }
    return results
# Defines a function that returns a dataframe containing the results for all the fitted models.
def finalResults(results):
    cols = ['Accuracy_score','Precision_score','Recall_score','F_score','Predictions','True positives','False positives','False negatives','True negatives']
    final_results = pd.DataFrame(results,cols).transpose()
    final_results = final_results.sort_values('F_score',ascending=False)
    return final_results


# Provided to give you a starting point. Try a variety of classifiers.

# Step 1: Split data
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print '\nInitial estimation of classifiers'
# a) Naive Bayes
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
NB_results = modelEvaluation(pred, labels_test)
#cross_val_score(clf,features_train,labels_train,cv=10,scoring='f1').mean()

# b) Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
DT_results = modelEvaluation(pred, labels_test)
#cross_val_score(clf,features_train,labels_train,cv=10,scoring='f1').mean()

#c) Logit Regression
from sklearn.linear_model import LogisticRegression
clf_logit = LogisticRegression()
clf_logit.fit(features_train,labels_train)
pred = clf_logit.predict(features_test)
LogitResults = modelEvaluation(pred, labels_test)
#cross_val_score(clf,features_train,labels_train,cv=10,scoring='f1').mean()

#d) K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
K_NNResults = modelEvaluation(pred, labels_test)
#cross_val_score(clf,features_train,labels_train,cv=10,scoring='f1').mean()

#Results for initial model estimations
print '\nResults for initial model estimations'
results = {'Naive Bayes': NB_results, 'Decision Tree': DT_results, 'Logit': LogitResults, 'KNN': K_NNResults}
print finalResults(results)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

# a) Naive Bayes
print '\na) Naive Bayes'
# Step 1: Split the data in training and testing set(see above)

# Step 2: Make pipeline
pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest',SelectKBest()),
        ('clf', GaussianNB())
    ])

# Step 3: Set parameters list
k_features = range(1,len(features_list))
param_grid = {'kbest__k': k_features}

#Step 4: Make cross validation iterator
#sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.1, train_size=None, random_state=42)
# Stratified KFold set to 10 folds

# Step 5: Make the GridSearchCV object clf with the paramters set above: pipeline, param_grid, sss
gridNB = GridSearchCV(estimator = pipeline,
                             param_grid = param_grid,
                             scoring = 'f1',
                             cv= 10)

# Step 6: Do training and cross validation with clf (fit to the data)
gridNB.fit(features_train,labels_train)
#pprint.pprint(grid.cv_results_) 


# Step 6.1: Get the set of best parameters and features
print '\nBest parameters chosen from the grid:'
pprint.pprint(gridNB.best_params_)
features_selected = [features_list[i+1] for i in gridNB.best_estimator_.named_steps['kbest'].get_support(indices=True)]
print 'The Features Selected by SKB - GS:'
pprint.pprint(features_selected)

# K best scores NB
f = features_list[1:]
s = gridNB.best_estimator_.named_steps['kbest'].scores_
results = {'features':f,'scores': s}
selectKBestScores = pd.DataFrame(results).sort_values('scores',ascending=False)
sns.barplot(x=selectKBestScores.features, y=selectKBestScores.scores)
plt.title("K Best Scores in Naive Bayes")
plt.xticks(rotation=90)
plt.show()

# Step 6.2: Get best estimator
clf = gridNB.best_estimator_

# Step 7: Predict with testing set
pred = clf.predict(features_test)
tunedNaiveBayes = modelEvaluation(pred, labels_test)
#test_classifier(clf, my_dataset, features_list)


# b) Decision Tree
print '\nb) Decision Tree'
# Step 1: Split the data in training and testing set(see above)

# Step 2: Make pipeline
pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest',SelectKBest()),
        ('clf', DecisionTreeClassifier())
        ])

# Step 3: Set parameters list
param_grid = {'kbest__k': k_features,
              'clf__splitter' : ['best','random'], 
              'clf__min_samples_split':[2,3]}

#Step 4: Make cross validation iterator
#sss = StratifiedShuffleSplit(n_splits=20, test_size=0.1, train_size=None, random_state=None)

# Step 5: Make the GridSearchCV object clf with the paramters set above: pipeline, param_grid, sss
gridDT = GridSearchCV(estimator = pipeline,
                             param_grid = param_grid,
                             scoring = 'f1',
                             cv= 10)

# Step 6: Do training and cross validation with clf (fit to the data)
gridDT.fit(features_train,labels_train)

# Step 6.1: Get the set of best parameters
print '\nBest parameters chosen from the grid:'
pprint.pprint(gridDT.best_params_)

# Step 6.2: Get best estimator feature importances
features_selected = [features_list[i+1] for i in gridDT.best_estimator_.named_steps['kbest'].get_support(indices=True)]
print 'The Features Selected by SKB - GS:'
pprint.pprint(features_selected)

f = features_list[1:]
s = gridDT.best_estimator_.named_steps['kbest'].scores_
results = {'features':f,'scores': s}
selectKBestScores = pd.DataFrame(results).sort_values('scores',ascending=False)
sns.barplot(x=selectKBestScores.features, y=selectKBestScores.scores)
plt.title("K Best Scores in Decision Tree")
plt.xticks(rotation=90)
plt.show()

# Step 6.3: Get best estimator
clf_final = gridDT.best_estimator_

# Step 7: Predict with testing set
pred = clf.predict(features_test)
tunedDecisionTree = modelEvaluation(pred, labels_test)
#test_classifier(clf_final, my_dataset, features_list)


# c) Logit Regression
print '\nc) Logit Regression'
# Step 1: Split the data in training and testing set(see above)

# Step 2: Make pipeline
pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest',SelectKBest()),
        ('clf', LogisticRegression())
        ])

# Step 3: Set parameters list
param_grid = {'kbest__k': k_features,
              'clf__C' : [1,1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9,2], 
              'clf__solver':['liblinear','sag']}

#Step 4: Make cross validation iterator
#sss = StratifiedShuffleSplit(n_splits=20, test_size=0.1, train_size=None, random_state=None)

# Step 5: Make the GridSearchCV object clf with the paramters set above: pipeline, param_grid, sss
gridLogit = GridSearchCV(estimator = pipeline,
                             param_grid = param_grid,
                             scoring = 'f1',
                             cv= 10)

# Step 6: Do training and cross validation with clf (fit to the data)
gridLogit.fit(features_train,labels_train)

# Step 6.1: Get the set of best parameters
print '\nBest parameters chosen from the grid:'
pprint.pprint(gridLogit.best_params_)

# Step 6.2: Get best estimator
features_selected = [features_list[i+1] for i in gridLogit.best_estimator_.named_steps['kbest'].get_support(indices=True)]
print 'The Features Selected by SKB - GS:'
pprint.pprint(features_selected)

f = features_list[1:]
s = gridLogit.best_estimator_.named_steps['kbest'].scores_
results = {'features':f,'scores': s}
selectKBestScores = pd.DataFrame(results).sort_values('scores',ascending=False)
sns.barplot(x=selectKBestScores.features, y=selectKBestScores.scores)
plt.title("K Best Scores in Logistic Regression")
plt.xticks(rotation=90)
plt.show()

# Step 6.3: Get best estimator
clf = gridLogit.best_estimator_

# Step 7: Predict with testing set
pred = clf.predict(features_test)
tunedLogit = modelEvaluation(pred, labels_test)
#test_classifier(clf, my_dataset, features_list)


# d) K-Nearest Neighbors
print '\nd) K-Nearest Neighbors'
# Step 1: Split the data in training and testing set(see above)

# Step 2: Make pipeline
pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest',SelectKBest()),
        ('clf', KNeighborsClassifier())
        ])
# Step 3: Set parameters list
param_grid = {'kbest__k': k_features,
              'clf__n_neighbors' : [4,5,6,7], 
              'clf__weights':['uniform','distance']}

#Step 4: Make cross validation iterator
#sss = StratifiedShuffleSplit(n_splits=20, test_size=0.1, train_size=None, random_state=None)

# Step 5: Make the GridSearchCV object clf with the paramters set above: pipeline, param_grid, sss
gridKNN = GridSearchCV(estimator = pipeline,
                             param_grid = param_grid,
                             scoring = 'f1',
                             cv= 10)

# Step 6: Do training and cross validation with clf (fit to the data)
gridKNN.fit(features_train,labels_train)

f = features_list[1:]
s = gridKNN.best_estimator_.named_steps['kbest'].scores_
results = {'features':f,'scores': s}
selectKBestScores = pd.DataFrame(results).sort_values('scores',ascending=False)
sns.barplot(x=selectKBestScores.features, y=selectKBestScores.scores)
plt.title("K Best Scores in K-NN classifier")
plt.xticks(rotation=90)
plt.show()

# Step 6.1: Get the set of best parameters
print '\nBest parameters chosen from the grid:'
pprint.pprint(gridKNN.best_params_)

# Step 6.2: Get best estimator
clf = gridKNN.best_estimator_

# Step 7: Predict with testing set
pred = clf.predict(features_test)
tunedKNN = modelEvaluation(pred, labels_test)
#test_classifier(clf, my_dataset, features_list)

#Final results for fine-tuned algorithms
print '\nFinal results for fine-tuned algorithms'
results = {'Naive Bayes': tunedNaiveBayes, 'Decision Tree': tunedDecisionTree, 'Logit': tunedLogit, 'KNN': tunedKNN}
print finalResults(results)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# The dumped classifier is the Decision Tree
dump_classifier_and_data(clf_final, my_dataset, features_list)

print "The end"