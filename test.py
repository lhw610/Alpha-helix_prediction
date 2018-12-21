import sys
import os
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import getFeat
import metric

# Predicting on certain Uniprot from test set. A random UniProt was used to show the result
# Predict using the Random Forest model
# example with Uniprot "A0R4Q6"
uniprot = "A0R4Q6"

model.get_features = getFeat.get_features
model.predict_uniprot = metric.predict_uniprot
model.plotROC = metric.plotROC

# get the feature and label
feat,label = getFeat.get_features(uniprot, nargout = 2)

# get the prediction
pred = model.predict_uniprot(model, uniprot, random_forest)

# get the probability for each class
probas = model.predict_uniprot(model, uniprot, random_forest, nargout = 2)
probas = [x[1] for x in probas]

# Plot the ROC curve for random forest
plot = model.plotROC(np.array(label), probas)
plt.savefig('test_rf.png')
plt.plot
plt.figure()

# Predicting with Neural Network model

model2.get_features = getFeat.get_features
model2.predict_uniprot = metric.predict_uniprot
model2.plotROC = metric.plotROC

# get the feature and label
feat,label = getFeat.get_features(uniprot, nargout = 2)

# get the prediction
pred = model2.predict_uniprot(model2, uniprot, neural_network)

# get the probability for each class
probas = model2.predict_uniprot(model2, uniprot, neural_network, nargout = 2)
probas = [x[1] for x in probas]

# Plot the ROC curve for neural network
plot = model2.plotROC(np.array(label), probas)
plt.savefig('test_nn.png')
plt.plot
plt.figure()