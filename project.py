#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

os.chdir('/Users/groth/Desktop/2021-sjsu/2021-fall/245/project/')
plt.rcParams['figure.dpi'] = 300

f1 = {'addr_state', 'annual_inc', 'application_type', 'dti', 'emp_length', 
      'fico_range_high', 'fico_range_low',  'grade', 'home_ownership', 
      'initial_list_status', 'installment', 'int_rate', 'loan_amnt', 
      'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 
      'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 
      'sub_grade', 'term', 'title', 'total_acc', 'verification_status'}

f2 = {'acc_now_delinq', 'annual_inc', 'delinq_2yrs', 'dti', 'emp_length', 
      'fico_range_high', 'fico_range_low', 'home_ownership', 'il_util', 
      'installment', 'int_rate', 'loan_amnt', 'loan_status', 
      'max_bal_bc', 'mths_since_last_major_derog', 'num_accts_ever_120_pd', 
      'num_actv_rev_tl', 'num_rev_accts', 'num_sats', 'num_tl_90g_dpd_24m', 
      'open_act_il', 'pct_tl_nvr_dlq', 'pub_rec', 'pub_rec_bankruptcies', 
      'revol_util', 'sub_grade', 'tax_liens', 'tot_coll_amt', 
      'hardship_flag', 'hardship_status', 'hardship_length', 
      'debt_settlement_flag' }

f3 = {'addr_state', 'annual_inc', 'application_type', 'dti', 
      'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 
      'fico_range_low', 'grade', 'hardship_flag', 'home_ownership', 
      'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 
      'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 
      'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 
      'sub_grade', 'tax_liens', 'term', 'title', 'total_acc', 
      'verification_status'}

f4 = {'acc_now_delinq', 'acc_open_past_24mths', 'addr_state', 
      'annual_inc',  'all_util', 'bc_util', 'revol_util', 'dti', 
      'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 
      'home_ownership', 'inq_last_12m', 'int_rate', 'loan_amnt', 
      'mort_acc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 
      'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 
      'open_acc_6m', 'open_il_24m', 'open_rv_24m', 'percent_bc_gt_75', 
      'pub_rec', 'pub_rec_bankruptcies', 'term', 'total_acc', 
      'verification_status', 'loan_status'}

features = f1 | f2 | f3 | f4
to_remove = {'hardship_status', 'hardship_length', 'tot_coll_amt',
             'mths_since_last_major_derog', 'id', 'grade', 'emp_title',
             'issue_d', 'title', 'debt_settlement_flag', 'hardship_flag'}
to_use = features - to_remove

# ----------------------------------------------------------------------------
# Some Preprocessing

df = pd.read_csv('loandata/Loan2007-2020Q3.gzip', usecols = to_use)


# reformat int_rate and revol_util into float
pattern = r"\s{0,}(\d{1,}\.?\d{0,})\%"
replace = lambda x: x.group(1)
cols_to_replace = ['int_rate', 'revol_util']
for col in cols_to_replace:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(pat = pattern, repl = replace, 
                                      regex = True)
    df[col] = df[col].astype('float64')

# ensure dti > 0
# DO THIS BEFORE SEPARATING INTO NUMERICAL AND CATEGORICAL VARS
df.drop(index = df[df['dti'] < 0].index, 
              axis = 0, inplace = True)

# fico ranges -> fico score (average)
df['fico_score'] = df['fico_range_low'] + \
    df['fico_range_high'] // 2
df.drop(['fico_range_low', 'fico_range_high'], axis = 1, inplace = True)

# select features
# investigate features
# # check numerical variables for anomalies 
# transform categorical into numerical   <-----
# impute missing values
# check correlation of numerical features

# ----------------------------------------------------------------------------
# Numerical Variables

num_vars = df.columns[df.dtypes == 'float64']
d = df[num_vars].describe()

# detect cols with extreme outliers 
# check that the median isn't 0 (o/w the IQR calculation -> 0)
idx_series = d.apply(lambda x: x['50%'] != 0 and \
                x['max'] > (5 * x['75%'] - x['25%']), axis = 0)
cols_to_adjust = d.columns[idx_series.values == True]

# replace extreme outliers with 1.5 * (Q3 - Q1) -> mild outlier value
for col in cols_to_adjust:
    third_quartile = df[col].quantile(0.75)
    thresh = df[col].quantile(0.75) - df[col].quantile(0.25)
    df[col].where(cond = df[col] < 5 * thresh,
                        other = third_quartile + (1.5 * thresh), 
                        inplace = True, 
                        axis = 0)
# pd.set_option('max_columns', 38)
# df[num_vars].describe()

# alternatively, can find indexes for extreme outliers and delete those rows



# to_plot = num_vars.sample(frac = 0.05, random_state = 111)

# for col in num_vars.columns:
#     g = sns.histplot(data = to_plot, x = col)
#     g.get_xaxis().set_visible(False)
#     g.set_title(f'{col}')
#     plt.show()
#     g.clear()

# ----------------------------------------------------------------------------
# Categorical Variables
cat_vars = df.columns[df.dtypes == 'object']

# emp_length -> int
# see arg na_action; avoid applying to missing values & keep them as NaN
df['employment_len'] = df['emp_length'].map(
    {'< 1 year': 0,
     '1 year': 1,
     '2 years': 2,
     '3 years': 3,
     '4 years': 4,
     '5 years': 5,
     '6 years': 6,
     '7 years': 7,
     '8 years': 8,
     '9 years': 9,
     '10+ years': 10})
df.drop(['emp_length'], axis = 1, inplace = True)

# credit line -> year (STRING)
pattern = r"\w+\-(\d{4})"
replace = lambda x: x.group(1)
df['earliest_cr_line'] = df['earliest_cr_line'].str.replace(
    pat = pattern, repl = replace, regex = True)

# home ownership -> consolidate categories ANY, OTHER, NONE
df['home_status'] = df['home_ownership'].map(
    {'MORTGAGE': 'MORTGAGE',
     'RENT': 'RENT',
     'OWN': 'OWN',
     'OTHER': 'OTHER',
     'ANY': 'OTHER',
     'NONE': 'OTHER'})
df.drop(labels = ['home_ownership'], axis = 1, inplace = True)

# loan_status -> binary label (Paid Off: 0, Charged Off: 1)
df['y'] = df['loan_status']\
    .map({'Fully Paid': 0, 'Charged Off': 1})
df.dropna(subset = ['y'], axis = 0, inplace = True)
df.drop(['loan_status'], axis = 1, inplace = True)

# new_cat_vars = df.columns[df.dtypes == 'object']
# plot categorical vars with horizontal countplot
# for col in new_cat_vars:
#     o = df[col].value_counts().index
#     g = sns.countplot(data = df, y = col, order = o)
#     g.set_ylabel(f'{col}', fontsize = 9)
#     g.set_yticklabels(labels = o, rotation = 0, fontsize = 7)
#     plt.show()
#     g.clear()

# ----------------------------------------------------------------------------
# Missing Values

missing = df.isna().sum().apply(lambda x: x/df.shape[0] * 100)
print(missing[missing > 0].sort_values(ascending = False))

# proportion of missing values
# all_util                 45.506037
# mths_since_recent_inq    12.441885
# employment_len            6.299746
# bc_util                   3.695852
# percent_bc_gt_75          3.664943
# pct_tl_nvr_dlq            3.638120
# num_accts_ever_120_pd     3.629842
# num_tl_30dpd              3.629842
# num_tl_90g_dpd_24m        3.629842
# pub_rec_bankruptcies      0.037466
# tax_liens                 0.002096

cols_with_missing_vals = ['all_util', 'mths_since_recent_inq', 
                          'employment_len', 'bc_util',
                          'percent_bc_gt_75', 'pct_tl_nvr_dlq',
                          'num_accts_ever_120_pd', 'num_tl_30dpd',
                          'num_tl_90g_dpd_24m', 'pub_rec_bankruptcies',
                          'tax_liens']

for col in cols_with_missing_vals:
    g = sns.histplot(x = df[col])
    plt.show()
    g.clear()
    
# all_util: normally distributed
# mths_since_recent_inq:skewed to the right
# employment_len: variable, conservative transformation -> 0 yrs
# bc_util: variable
# percent_bc_gt_75: symmetric
# pct_tl_nvr_dlq: dirac delta skewed to the left
# num_accts_ever_120_pd: dirac delta skewed to the right
# num_tl_30dpd: dirac delta skewed to the right
# num_tl_90g_dpd_24m: dirac delta skewed to the right
# pub_rec_bankruptcies: dirac delta skewed to the right
# tax_liens: dirac delta skewed to the right

df.to_csv('loan-data-with-missing-vals.csv.gzip', 
          compression = 'gzip', index = None)


# impute values





# for now, just drop any rows with missing values
# temp = df.copy().dropna(axis = 0, how = 'any')





# ----------------------------------------------------------------
# Feature Evaluation (multi-colinearity/correlation)

# check correlation between variables
correlation = temp.corr(method = 'pearson')
sns.heatmap(correlation)



# ----------------------------------------------------------------
# Models

df = pd.read_csv('loan_cleaned.csv.gzip', compression = 'gzip')
y = pd.read_csv('loan_target.csv')




# DUMMY VARIABLES
# addr_state -> dummies
# application_type -> dummies (2 labels)
# initial_list_status -> dummies
# sub_grade -> dummies 
# term -> dummies (2 labels)
# verification status -> dummies (3 labels)
# purpose -> dummies (14 labels)
cols_to_dummify = ['addr_state', 'application_type', 'initial_list_status',
        'sub_grade', 'term', 'verification_status', 'purpose']
train = pd.get_dummies(data = df, columns = cols_to_dummify)

# scale features between [0, 1]
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train)
scaled_vars = pd.DataFrame(scaler.transform(train), 
                           columns = train.columns)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_vars, y, train_size = 0.8, random_state = 111, shuffle = True)

pca = PCA(n_components = 2, random_state = 111)
pca.fit(X_train)
pca_train = pca.transform(X_train)
pca_df = pd.DataFrame(pca_train, columns = ['C1', 'C2'])

sns.scatterplot(data = pca_df, x = 'C1', y = 'C2', 
                hue = y_train.values.reshape(-1),
                palette = 'winter')


def runAndEvaluateModel(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    cl_pred = classifier.predict(X_test)
    confused = confusion_matrix(y_true = y_test, y_pred = cl_pred)
    score = classifier.score(X_test, y_test)
    print(confused)
    # sns.heatmap(pd.DataFrame(confused))
    print(classification_report(y_true = y_test, y_pred = cl_pred))
    return (score, confused, classifier)



# change max_iter if it doesn't converge
lr = LogisticRegression(solver = 'saga', n_jobs = -1, max_iter = 500)
lr.fit(X_train, y_train.values.reshape(-1))
lr_pred = lr.predict(X_test)
lr.score(X_test, y_test) 

# class imbalance problem (high accuracy but low recall for class 1)
# axis = 1 -> recall
# axis = 0 -> precision
confused = confusion_matrix(y_true = y_test, y_pred = lr_pred)
print(confused)
print(classification_report(y_test, lr_pred))


# other components to try/check:
# feature selection
# ensemble
# imputation
# other linear models
# cross validation

# better than logistic regression for class 1 recall, worse on c1 precision
bag = BaggingClassifier(base_estimator = None,
                        n_estimators = 10,
                        max_samples = 100_000,
                        max_features = 1.0,
                        oob_score = True,
                        n_jobs = -1,
                        random_state = 111)
bag.fit(X_train, y_train.values.reshape(-1))
bag_pred = bag.predict(X_test)
bag.score(X_test, y_test)

confused = confusion_matrix(y_true = y_test, y_pred = bag_pred)
print(confused)
print(classification_report(y_test, bag_pred))


rf = RandomForestClassifier(
    n_estimators = 100,
    criterion = 'gini',
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = 'sqrt',
    bootstrap = True,
    oob_score = True,
    n_jobs = -1,
    random_state = 111,
    warm_start = True,
    max_samples = 100_000)
rf.fit(X_train, y_train.values.reshape(-1))
rf_pred = rf.predict(X_test)
rf.score(X_test, y_test)

confused = confusion_matrix(y_true = y_test, y_pred = rf_pred)
print(confused)
print(classification_report(y_test, rf_pred))




