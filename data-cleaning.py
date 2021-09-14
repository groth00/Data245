import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

plt.rcParams['figure.dpi'] = 300

# %%
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

# %%
# Some Preprocessing 

df = pd.read_csv('loandata/Loan2007-2020Q3.gzip', usecols = to_use)

# %%
# reformat int_rate and revol_util into float
pattern = r"\s{0,}(\d{1,}\.?\d{0,})\%"
replace = lambda x: x.group(1)
cols_to_replace = ['int_rate', 'revol_util']
for col in cols_to_replace:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(pat = pattern, repl = replace, 
                                      regex = True)
    df[col] = df[col].astype('float64')
    
# %%
# ensure dti > 0
df.drop(index = df[df['dti'] < 0].index, axis = 0, inplace = True)

# %%
# fico ranges -> fico score (average)
df['fico_score'] = df['fico_range_low'] + \
    df['fico_range_high'] // 2
df.drop(['fico_range_low', 'fico_range_high'], axis = 1, inplace = True)

# %%
# ----------------------------------------------------------------------------
# Numerical Variables

num_vars = df.columns[df.dtypes == 'float64']
d = df[num_vars].describe()

# %%
# detect cols with extreme outliers 
# check that the median isn't 0 (o/w the IQR calculation -> 0)
idx_series = d.apply(lambda x: x['50%'] != 0 and \
                x['max'] > (5 * x['75%'] - x['25%']), axis = 0)
cols_to_adjust = d.columns[idx_series.values == True]

# replace extreme outliers with 1.5 * (Q3 - Q1) -> mild outlier value
# alternatively, can find indexes for extreme outliers and delete those rows
for col in cols_to_adjust:
    third_quartile = df[col].quantile(0.75)
    thresh = df[col].quantile(0.75) - df[col].quantile(0.25)
    df[col].where(cond = df[col] < 5 * thresh,
                        other = third_quartile + (1.5 * thresh), 
                        inplace = True, 
                        axis = 0)

# %%
pd.set_option('max_columns', 38)
df[num_vars].describe()

# %%
to_plot = num_vars.sample(frac = 0.05, random_state = 111)

for col in num_vars.columns:
    g = sns.histplot(data = to_plot, x = col)
    g.get_xaxis().set_visible(False)
    g.set_title(f'{col}')
    plt.show()
    g.clear()

# %%
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

# %%
new_cat_vars = df.columns[df.dtypes == 'object']
# plot categorical vars with horizontal countplot
for col in new_cat_vars:
    o = df[col].value_counts().index
    g = sns.countplot(data = df, y = col, order = o)
    g.set_ylabel(f'{col}', fontsize = 9)
    g.set_yticklabels(labels = o, rotation = 0, fontsize = 7)
    plt.show()
    g.clear()

# %%

