# Alpha-helix_prediction
Generates the statistical features(mean and correlation from the surrounding protein) and train random forest and neural network model
to predict the secondary structure(Alpha helix)

# Instruction
# Training
Run feature_generator.py to read expasy amino acid scale and generates statistical features. This file generates
features for training and test set. Use train.py to train random forest and neural network classifer. It saves the classifier using pickle

# Test
Run test.py to get prediction.(uniprot "A0R4Q6" is set as default. Change the uniprot to the one that you want to predict)
It use ROC wiht AUC for evaluating the performance of random forest classifier and MLP classifier

# Citation
Using plotRoc function from Shayne Wierbowski(sdw95@cornell.edu)

