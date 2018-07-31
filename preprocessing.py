"""
Created on 14.07.18, 22:15
author: oskar
"""

###########
##Imports##
###########

#Basic libraries
import numpy as np
import pandas as pd

#Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

#System
from os import path


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


def numerate_pymnt_d(value):
    translator = dict(zip(['Jan',
                           'Feb',
                           'Mar',
                           'Apr',
                           'May',
                           'Jun',
                           'Jul',
                           'Aug',
                           'Sep',
                           'Oct',
                           'Nov',
                           'Dec'],
                          range(1,13)))
    if pd.isnull(value):
        return value
    else:
        return translator[value[:3]] + 12*int(value[-4:])


def numerate_emp_length(value):
    if value == '< 1 year':
        return 0
    elif value == '10+ years':
        return 10
    elif pd.isnull(value):
        return np.nan
    else:
        return int(value.split()[0])


def input(X, y, model=None):
    if model:
        clf = model
    else:
        clf = KNeighborsRegressor(n_neighbors=100)

    no_nan_indeces = y.dropna().index
    nan_indeces = y[y.isna()].index

    clf.fit(X.loc[no_nan_indeces],
            y[no_nan_indeces])

    inputed_values = clf.predict(X.loc[nan_indeces])

    return inputed_values


def full_input_by(df, by_cols, to_col, model=None):
    def input(X, y, model):
        no_nan_indeces = y.dropna().index
        nan_indeces = y[y.isna()].index

        model.fit(X.loc[no_nan_indeces],
                y[no_nan_indeces])

        inputed_values = model.predict(X.loc[nan_indeces])

        return inputed_values

    output = df.copy()

    useful_idx = output[by_cols].dropna().index
    to_input = output[to_col][useful_idx]
    if model:
        clf = model
    else:
        clf = KNeighborsRegressor(n_neighbors=100)

    nan_indeces = output.loc[useful_idx][output[to_col][useful_idx].isnull()].index

    output[to_col][nan_indeces] = input(output[by_cols].loc[useful_idx],
                                        to_input,
                                        clf)

    return output

def load_data(path, subsampling=None, seed=17):
    if subsampling:
        data = pd.read_csv(path).sample(n=subsampling, random_state=seed)

    else:
        data = raw_df = pd.read_csv(path)

    return data


def binarize(X, columns):
    binarizer = LabelBinarizer()
    output = X.copy()
    for col in columns:
        binarizer.fit(X[col])
        binned_names = [col + '_' + str(x) for x in binarizer.classes_]
        binned_values = np.transpose(binarizer.transform(output[col]))
        d = dict(zip(binned_names, binned_values))
        output = output.assign(**d)
        output.drop(col, inplace=True, axis=1)

    return output


def encode(X):
    encoder = LabelEncoder()
    output = X.copy()
    for col in output.columns:
        encoder.fit(output[col])
        output[col] = encoder.transform(output[col])

    return output


def main():
    #########################
    ##Step 1: Load the data##
    #########################
    directory = r"C:\Users\duos8001\Documents\Python\Machine Learning tutorial\Default Prediction\Data"
    filename = 'loan.csv'
    full_path = path.join(directory, filename)
    n = 800000

    raw_df = load_data(full_path, n)
    df = raw_df.copy()
    print("Data loaded. Number of entries: ", len(raw_df))

    ###############################
    ##Step 2: Preprocess the data##
    ###############################
    df['is_default'] = df.apply(is_default, axis=1)
    df['title'] = df.apply(group_title_column, axis=1)
    df['emp_length'] = df['emp_length'].apply(numerate_emp_length)
    df['last_pymnt_d'] = df['last_pymnt_d'].apply(numerate_pymnt_d)
    df['next_pymnt_d'] = df['next_pymnt_d'].apply(numerate_pymnt_d)
    not_nan_idx = df['emp_title'].dropna().index
    df['emp_title'][not_nan_idx] = df['emp_title'][not_nan_idx].apply(lambda x: x.lower())
    df['emp_title'].fillna('other', inplace=True)
    corrs = df.corr()

    # input_by = ['member_id', 'out_prncp_inv']
    # X = df[input_by]
    # to_input = df['last_pymnt_d']
    # nan_indeces = df[df['last_pymnt_d'].isnull()].index
    # df['last_pymnt_d'][nan_indeces] = input_values(df[input_by],
    #                                                to_input)
    # input_by = ['last_pymnt_d']
    # X = df[input_by]
    # to_input = df['next_pymnt_d']
    # nan_indeces = df[df['next_pymnt_d'].isnull()].index
    # df['last_pymnt_d'][nan_indeces] = input_values(df[input_by],
    #                                                to_input,
    #                                                LinearRegression())
    # input_by = ['revol_bal']
    # useful_idx = df[input_by].dropna().index
    # to_input = df['total_rev_hi_lim'][useful_idx]
    # nan_indeces = df.loc[useful_idx][df['total_rev_hi_lim'][useful_idx].isnull()].index
    # df['total_rev_hi_lim'][nan_indeces] = input_values(df[input_by].loc[useful_idx],
    #                                                    to_input,
    #                                              LinearRegression())
    #
    # input_by = ['annual_inc', 'revol_bal', 'total_rev_hi_lim','funded_amnt_inv']
    # useful_idx = df[input_by].dropna().index
    # to_input = df['tot_cur_bal'][useful_idx]
    # nan_indeces = df.loc[useful_idx][df['tot_cur_bal'][useful_idx].isnull()].index
    # df['tot_cur_bal'][nan_indeces] = input_values(df[input_by].loc[useful_idx],
    #                                              to_input)
    #
    #
    # to_input_col = 'emp_length'
    # df = full_input_by(df, to_input_col, input_by)
    #
    # input_by = ['funded_amnt_inv', 'loan_amnt', 'funded_amnt', 'total_acc', 'funded_amnt']
    # useful_idx = df[input_by].dropna().index
    # to_input = df['emp_length'][useful_idx]
    # nan_indeces = df.loc[useful_idx][df['emp_length'][useful_idx].isnull()].index
    # df['emp_length'][nan_indeces] = input_values(df[input_by].loc[useful_idx],
    #                                              to_input)

    inputation = [['last_pymnt_d',['member_id', 'out_prncp_inv'],None],
                  ['next_pymnt_d',['last_pymnt_d'],LinearRegression()],
                  ['total_rev_hi_lim',['revol_bal'],LinearRegression()],
                  ['tot_cur_bal',['annual_inc', 'revol_bal', 'total_rev_hi_lim', 'funded_amnt_inv'],None],
                  ['emp_length', ['funded_amnt_inv', 'loan_amnt', 'funded_amnt', 'total_acc', 'funded_amnt'], None]]

    for [to_input_col, input_by_col, model] in inputation:
        df = full_input_by(df, input_by_col, to_input_col, model)

    df.drop(labels = df[df.revol_util.isnull()].index, inplace=True)
    df = df.dropna(axis=1)

    y = df['is_default']
    X = df.drop(['is_default', 'loan_status'], axis=1)
    X = encode(X)

    cols_to_binarize = [x for x in X.columns if len(X[x].value_counts())<10]
    X = binarize(X, cols_to_binarize)
    print("Data preprocessing finished")

    return X, y