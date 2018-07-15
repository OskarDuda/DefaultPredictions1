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

#Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#ML
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder

#System
from os import path

########################
##Function definitions##
########################
def is_default(line):
    values_perceived_as_default = ['Charged Off',
                                   'Default',
                                   'Does not meet the credit policy. Status:Charged Off',
                                   'Late (31-120 days)']

    return line['loan_status'] in values_perceived_as_default

def group_title_column(line):
    values_preserved = ['Debt consolidation',
                        'Credit card refinancing',
                        'Home improvement',
                        'Other',
                        'Debt Consolidation',
                        'Major purchase']

    if line['title'] not in values_preserved:
        return 'Other'
    else:
        return line['title']

#########################
##Step 1: Load the data##
#########################
directory = r"/home/oskar/PycharmProjects/McKinsey predictions/lending-club-loan-data"
filename = 'loan.csv'
full_path = path.join(directory, filename)
n = 12000
if n:
    df = pd.read_csv(full_path).sample(n=n)
else:
    df = pd.read_csv(full_path)
print("Data loaded. Number of entries: ", len(df))

###############################
##Step 2: Preprocess the data##
###############################
accepted_nans_per_column = 0.01
criteria = df.isna().sum() < len(df) * accepted_nans_per_column
df = df[criteria.index[criteria]]

df['is_default'] = df.apply(is_default, axis=1)
df['title'] = df.apply(group_title_column, axis=1)
df = df.dropna()

# encoder = LabelEncoder()
y = df['is_default']
X = df.drop(['is_default', 'loan_status'], axis=1)
# encoder.transform(X)
print("Data transformed")

####################
##Step 3: Plotting##
####################
# plt.cla()
# sns.pairplot(X[X.columns[11:21]].join(y), hue = 'is_default')
# sns.heatmap(df[df.columns[:11]].isna())
# plt.show()

##########################
##Step 4: Training model##
##########################
best_separators = ['id', 'member_id', 'delinq_2yrs', 'pub_rec', 'annual_inc', 'total_rec_prncp', 'last_pymnt_amnt']
X = df[best_separators]
y = df['is_default']


basic_model = LogisticRegression()
model = basic_model
model = BaggingClassifier(base_estimator=basic_model, n_estimators=5000, verbose=1, max_samples=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Accuracy of trained model without feature space transformation is {:.2f}%".format(100*score))

# red = PCA(n_components=len(X.columns))
# red = LinearDiscriminantAnalysis(n_components=len(X.columns))
# red.fit(X,y)
# X_r = red.transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.2)
# model.fit(X_train, y_train)
# score = model.score(X_test, y_test)
# print("Accuracy of trained model with feature space transformation is {:.2f}%".format(100*score))
