"""
Created on 14.07.18, 22:15
author: oskar
"""

###########
##Imports##
###########
#Basic libraries
import pandas as pd
import numpy as np
import time

#Sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_validate

#Local libraries
import preprocessing as prp


def main(X, y):
    model_selector = 'rand_forest'
    reductor_selector = 'none'
    validation_set_size = 0.8
    seed = 17

    models = {'rand_forest' : RandomForestClassifier(n_estimators=500, verbose=2, max_depth=4, max_features=50, random_state=seed),
              'boosted_trees' : GradientBoostingClassifier(n_estimators=1600, verbose=2, max_features=40, subsample=0.5, random_state=seed)}
    model = models.get(model_selector)
    reductors = {'pca' : PCA(n_components=len(X.columns), random_state=seed),
                 'lda' : LinearDiscriminantAnalysis(n_components=len(X.columns)),
                 'none' : None}
    red = reductors.get(reductor_selector)

    if red:
        print("Feature space transformation")
        red.fit(X,y)
        X_r = red.transform(X)
        msg = lambda x: "Accuracy of trained model with feature space transformation is {:.4f}%".format(100*x)
    else:
        X_r = X
        msg = lambda x: "Accuracy of trained model without feature space transformation is {:.4f}%".format(100*x)

    print("Model selected. Starting training {}".format(time.asctime()))
    X_train, X_test, y_train, y_test = train_test_split(X_r, y, train_size=validation_set_size, random_state=seed)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(msg(score))
    return score


if __name__ == '__main__':
    X, y = prp.main()
    score = main(X, y)