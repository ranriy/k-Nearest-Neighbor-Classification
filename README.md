# k-Nearest-Neighbor-Classification
Classified NBA players into 5 positions (classes) on the basketball court using k-NN model. Performed 10-fold stratified cross-validation.

# Dataset: 
NBAstats.csv file

# Applied k-NN algorithm using scikit-learn:
1) Divided the data into a training set and a test set.
2) Instantiated the KNeighborsClassifier class with weights as inverse of the distance and number of neighbors as 7.
3) Used fit method to fit the classifier using the training set.
4) Used predict method to make predictions on the test data.For each data point in the test set, this computes its nearest neighbors in the training set.
5) Evaluated the model on the test set using the score method, which for classification computes the fraction of correctly classified samples.
6) Used 10-fold stratified cross-validation. Observed the accuracy of each fold and the average accuracy across all folds.
