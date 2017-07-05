import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

original_headers = list(nba.columns.values)

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

feature_columns = ['FG%','3P%','2P%','FT%','ORB','DRB','AST','STL','BLK','TOV','PF']


# column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

# normalizing the values in the feature columns.
# When values are scaled between 0 to 1, all of the selected features contribute equally to distance computation
normalized_nba_feature = preprocessing.normalize(nba_feature)

train_feature, test_feature, train_class, test_class = \
    train_test_split(normalized_nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25) #75/25 data split

training_accuracy = []
test_accuracy = []

#7NN classifier with w=1/d
knn = KNeighborsClassifier(n_neighbors=7, weights='distance',  metric='minkowski', p=1)
knn.fit(train_feature, train_class)
prediction = knn.predict(test_feature)

print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))


print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

scores = cross_val_score(knn, normalized_nba_feature, nba_class, cv=10) #cv=10 for 10 folds
print("Cross-validation scores for each folds: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
