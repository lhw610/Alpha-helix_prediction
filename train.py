import sys
import os
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import getFeat

# loading all the necessary elements
features = pd.read_csv("train_features.csv")
labels = pd.read_csv("Training_Labels.txt", '\t')

# Split the data into the train and validation set
data_split = np.split(features, [1907], axis=0)
training_label = data_split[1]['Label'].tolist()
training_data = data_split[1].drop(columns = ["Label","UniProt",'Position','AA'])
validation_label = data_split[0]['Label'].tolist()
validation_data = data_split[0].drop(columns = ["Label","UniProt",'Position','AA'])

# Training the random forest
random_forest = []
for i in range(100):
    model = tree.DecisionTreeClassifier(max_depth=3, splitter="random")
    model.fit(training_data, training_label)
    random_forest.append(model)

# save classifier
import pickle
clf1 = pickle.dumps(model)
out = open("final_clf1.pickle", "w+")
out.write(clf1)
out.close()

# This time using neural network
# If early stopping set to true, it will automatically set aside 10% of training data as validation and terminate 
# training when validation score is not improving 
# It automatically divides into train and valida dataset. 
# so I don't need to check evey validation ROC curves manualy. 
neural_network = []
for i in range(10):
    model2 = MLPClassifier(activation='relu', early_stopping=True)
    model2.fit(training_data, training_label)
    neural_network.append(model2)

# save classifier
clf2 = pickle.dumps(model2)
out = open("final_clf2.pickle", "w+")
out.write(clf2)
out.close()