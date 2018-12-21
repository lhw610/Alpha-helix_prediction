import sys
import os
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import getFeat

# Read the basic information of protein and labels
features = pd.read_csv("Expasy_AA_Scales.txt", "\t")
labels = pd.read_csv("Training_Labels.txt", '\t')
features_all = labels.join(features.set_index("Amino Acid"), on="AA", how="inner") # match with labels
training_label = features_all[['Label']].copy()
features_all = features_all.sort_index()

# Now add my new features to the original features
col_list = features_all.columns.get_values()
my_feat_mean, my_feat_corr = getFeat.create_feat(features_all,col_list) # create the features using the function

#change to array 
my_feat_mean = np.array(my_feat_mean)
my_feat_corr = np.array(my_feat_corr)
training_data = features_all.copy()

# add mean and correlation features
for i in range(len(my_feat_mean[0,0,:])):
    training_data['myfeat{0}'.format(i)] = my_feat_mean[:,0,i]
for i in range(len(my_feat_corr[0,0,:])):
    training_data['myfeat{0}'.format(i+len(my_feat_mean[0,0,:]))] = my_feat_corr[:,0,i]
    
# remove origial scale featrues    
training_data = training_data.drop(columns = col_list)

# feature selection using Linear Support Vector Classification with L1 penalty
# this takes some time

training_label = features_all['Label'].tolist()
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=500).fit(training_data, training_label)
model = SelectFromModel(lsvc, prefit=True)
feat_new = model.transform(training_data)

# make training data using the selected features
feat_chosen = [] # list to store the feature chosen after feature selection
for col in training_data:
    for col2 in range(feat_new.shape[1]):
        if (np.array(training_data[col]) == feat_new[:,col2]).all():
            feat_chosen.append(col)
train_data = training_data[feat_chosen].copy()

# Add the uniprot, position, and AA again to features
train_data['UniProt']= features_all['UniProt']
train_data['Position']= features_all['Position']
train_data['AA']= features_all['AA']
train_data['Label']= features_all['Label']

# Export Features to CSV file
train_data.to_csv('train_features.csv',index=False)

# # Test feature construction

# Now, make Testing features same way I did with training features
# call all the necessary files

features = pd.read_csv("Expasy_AA_Scales.txt", "\t")
labels = pd.read_csv("Testing_Labels.txt", '\t')
features_all = labels.join(features.set_index("Amino Acid"), on="AA", how="inner")
testing_label = features_all[['Label']].copy()
features_all = features_all.sort_index()

# Now make features like the training features
col_list = features_all.columns.get_values()
my_feat_mean, my_feat_corr = getFeat.create_feat(features_all, col_list)
my_feat_mean = np.array(my_feat_mean)
my_feat_corr = np.array(my_feat_corr)
testing_data = features_all.copy()

# add mean and correlation features
for i in range(len(my_feat_mean[0,0,:])):
    testing_data['myfeat{0}'.format(i)] = my_feat_mean[:,0,i]
for i in range(len(my_feat_corr[0,0,:])):
    testing_data['myfeat{0}'.format(i+len(my_feat_mean[0,0,:]))] = my_feat_corr[:,0,i]
    
# remove origial scale featrues 
testing_data = testing_data.drop(columns = col_list)

# make testing data using the selected features from feature selection algorithm
test_data = testing_data[feat_chosen].copy()

# Add the uniprot, position, and AA again to features
test_data['UniProt']= features_all['UniProt']
test_data['Position']= features_all['Position']
test_data['AA']= features_all['AA']
test_data['Label']= features_all['Label']
# Export Features to CSV file
test_data.to_csv('test_features.csv',index=False)