import sys
import os
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt

# Read the features and labels
features = pd.read_csv("Expasy_AA_Scales.txt", "\t")
labels = pd.read_csv("Training_Labels.txt", '\t')
features_all = labels.join(features.set_index("Amino Acid"), on="AA", how="inner") # match with labels
training_label = features_all[['Label']].copy()
features_all = features_all.sort_index()

# construct my features using the average of surrounding residule for each features
# This function computes the mean and correlation between neighboring AA.
# These are great metric since the data is sequence data which means that relationship with
# previous, present, and after is very important.

def create_feat(features_all, col_list):
    features = features_all.drop(columns = col_list[:4])
    features = features.drop(columns = col_list[24:])
    training_data_arr = np.array(features)
    my_feat_mean = []
    my_feat_corr = []
    
    # compute mean with 4 neighboring AA
    for i in range(len(training_data_arr)):
        temp = []
        if i == 0:
            temp.append((training_data_arr[i][:]
                +training_data_arr[i+1][:])/2)
        elif i == 1 or i == len(training_data_arr)-2:
            temp.append((training_data_arr[i-1][:]+training_data_arr[i][:]
                +training_data_arr[i+1][:])/3)
        elif i == len(training_data_arr)-1:
            temp.append((training_data_arr[i-1][:]
                +training_data_arr[i][:])/2)
        else:
            temp.append((training_data_arr[i][:]+training_data_arr[i-1][:]+training_data_arr[i-2][:]
                +training_data_arr[i+1][:]+training_data_arr[i+2][:])/5)
        my_feat_mean.append(temp)

    # compute correlation with 2 neighboring AA
    for i in range(len(training_data_arr)):
        temp = []
        if i == 0:
            temp.append(np.correlate(training_data_arr[i][:]
                ,training_data_arr[i+1][:], 'same'))
        elif i == len(training_data_arr)-1:
            temp.append(np.correlate(training_data_arr[i-1][:]
                ,training_data_arr[i][:],'same'))
        else:
            temp.append((np.correlate(training_data_arr[i][:],training_data_arr[i-1][:], 'same')
                         +np.correlate(training_data_arr[i][:],training_data_arr[i+1][:], 'same'))/2)
        my_feat_corr.append(temp)
    return my_feat_mean, my_feat_corr


# This function constuct the features for the given Uniprot from the CSV file
# It has two output option. One for the features only and other for feature and label
def get_features(UniProt, nargout = 1):
    test_feat = pd.read_csv("test_features.csv")
    feat = test_feat.loc[test_feat['UniProt'] == UniProt]
    label = feat['Label'].tolist()
    feat = feat.drop(columns = ["Label","UniProt",'Position','AA'])
    if nargout == 1:
        return feat
    else:
        return feat, label