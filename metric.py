import sys
import os
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt

# for predicting probability
def predict_proba(forest, X):
    predictions = []
    for clf in forest:
        # Can either use probabilities from individual trees
        predictions.append(clf.predict_proba(X)[:, 1])
        
        # Or predictions from individual trees
        ##predictions.append(clf.predict(X))
    predictions = np.array(predictions)
    
    probas = []
    for i in range(len(predictions[0])):
        prob_1 = sum(predictions[:, i]) / float(len(predictions[:, i]))
        prob_0 =  1 - prob_1
        probas.append([prob_0, prob_1])
    return probas

# This function predicts the uniprot based on probabilty.
# It also returns the probability when nargout is set to anynumber other than 0
# It is basically one function that can be used as predicct_uniprot and predict_uniprot_proba

def predict_uniprot(self, UniProt, learning_model, nargout = 1):
    X = get_features(UniProt)
    if nargout == 1:
        return [int(x[1] >= 0.5) for x in predict_proba(learning_model, X)]
    else:
        return predict_proba(learning_model, X)

# Using Shyane's plot function
# Generates a ROC Curve for a given set of labels and scores
# - Accepts as parameters...
#   @labels - The label values
#   @scores - The predicted scores
#   @forcePosAUC - A boolean flag indicating whether scores should be inverted to maximize AUC
#                        (i.e. disallow an AUC < 0.50)
# - Returns the figure object, auc value, false positive rate, true positive rate, and
#    a boolean flad indicating whether or not scores were inverted
# - Written for general utility purposes 04/03/17 by Shayne Wierbowski
def plotROC(labels, scores, forcePosAUC=False):

    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    roc_auc = metrics.auc(fpr, tpr)

    inverted = False
    if(forcePosAUC and roc_auc < 0.5):
        fpr, tpr, _ = metrics.roc_curve(labels, -1*np.array(scores))
        roc_auc = auc(fpr, tpr)
        inverted = True

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange",
    lw=lw, label="ROC curve (area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if(inverted):
        plt.title("Receiver Operating Characteristic (Inversed Score)")
    else:
        plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    if(forcePosAUC):
        return fig, roc_auc, fpr, tpr, inverted
    else:
        return fig, roc_auc, fpr, tpr
# FUNCTION END