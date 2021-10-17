import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, MissingIndicator

plt.rcParams['figure.dpi'] = 300

# %% Picking features
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

# %% Load csv file
df = pd.read_csv('loandata/Loan2007-2020Q3.gzip', usecols = to_use)

# %% Reformat int_rate and revol_util into float
pattern = r"\s{0,}(\d{1,}\.?\d{0,})\%"
replace = lambda x: x.group(1)
cols_to_replace = ['int_rate', 'revol_util']
for col in cols_to_replace:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(pat = pattern, repl = replace, 
                                      regex = True)
    df[col] = df[col].astype('float64')
    
# %% Ensure dti > 0
df.drop(index = df[df['dti'] < 0].index, axis = 0, inplace = True)

# %% Fico ranges -> fico score (average)
df['fico_score'] = df['fico_range_low'] + \
    df['fico_range_high'] // 2
df.drop(['fico_range_low', 'fico_range_high'], axis = 1, inplace = True)

# %% Check numerical variables
num_vars = df.columns[df.dtypes == 'float64']
d = df[num_vars].describe()

# %% Replace outliers
# find cols with extreme outliers 
idx_series = d.apply(lambda x: x['50%'] != 0 and \
                x['max'] > (5 * x['75%'] - x['25%']), axis = 0)
cols_to_adjust = d.columns[idx_series.values == True]

# remove rows with extreme outliers: 10 * (Q3 - Q1)
for col in cols_to_adjust:
    thresh = df[col].quantile(0.75) - df[col].quantile(0.25)
    to_remove = df[df[col] > 10 * thresh].index
    df.drop(to_remove, axis = 0, inplace = True)

# %% Check result of outlier removal
pd.set_option('max_columns', 38)
print(df[num_vars].describe())

# %% Categorical Variables
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

# credit line -> year 
pattern = r"\w+\-(\d{4})"
replace = lambda x: x.group(1)
df['earliest_cr_line'] = df['earliest_cr_line'].str.replace(
    pat = pattern, repl = replace, regex = True)
df['earliest_cr_line'] = df['earliest_cr_line'].astype('float64')

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

# %% Plot numerical vars
to_plot = df.sample(frac = 0.05, random_state = 111)
tp_num_vars = to_plot.select_dtypes('float64')
y_to_plot = to_plot['y'].astype(str)
for col in tp_num_vars.columns:
    ax = plt.subplot()
    sns.boxplot(data = tp_num_vars, x = col, y = y_to_plot, ax = ax)
    ax.set_title(f'{col}')
    plt.show()
    
# %% Plot categorical vars with countplot
tp_cat_vars = to_plot.select_dtypes('object')
for col in tp_cat_vars.columns:
    ax = plt.subplot()
    o = to_plot[col].value_counts().index
    sns.countplot(data = tp_cat_vars, y = col, order = o, 
                  hue = y_to_plot, ax = ax)
    ax.set_ylabel(f'{col}', fontsize = 9)
    ax.set_yticklabels(labels = o, rotation = 0, fontsize = 7)
    ax.legend(loc = 'lower right')
    plt.show()
    
# %% Deal with missing values
missing = df.isna().sum().apply(lambda x: x/df.shape[0] * 100)
print(missing[missing > 0].sort_values(ascending = False))

# %% Plot vars with missing values
cols_with_missing_vals = missing[missing > 0]\
    .sort_values(ascending = False).index

for col in cols_with_missing_vals:
    g = sns.histplot(x = to_plot[col])
    plt.show()
    g.clear()

# %% Imputing all_util with mean
# i = SimpleImputer(strategy = 'mean')
# df['all_util'] = i.fit_transform(df['all_util'].values.reshape(-1, 1))
# print(df['all_util'].isna().sum())

# %% MissingIndicator
m = MissingIndicator(features = 'missing-only')
m.fit(df)
indicators = m.transform(df)
col_names = [name + '_missing' for name in missing[missing > 0].index]
# this is a dataframe of booleans
ind = pd.DataFrame(indicators, columns = col_names)
# convert it to int (False (not missing) -> 0, True (missing) -> 1)
ind = ind.astype(int)

# %% Impute with IterativeImputer + DecisionTreeRegressor
e = DecisionTreeRegressor(max_features = 'sqrt', 
                          max_depth = 3, random_state = 111)
imp = IterativeImputer(estimator = e)

num_vars = df[df.columns[df.dtypes == 'float64']]

# impute based on existing numerical variables
# it takes too long when using all variables 
imp.fit(num_vars)
imputed = imp.transform(num_vars)

# %% Convert categorical -> numerical and rejoin data
X = pd.DataFrame(imputed, columns = num_vars.columns)

cat_vars = df[df.columns[df.dtypes == 'object']]
dummies = pd.get_dummies(cat_vars)

# separate y
y = X['y']
X.drop('y', axis = 1, inplace = True)

# had problems joining because of indexes so reset all of them
X.reset_index(drop = True, inplace = True)
dummies.reset_index(drop = True, inplace = True)

# join dummy variables with numerical variables
cleaned_data = X.join(dummies)

cleaned_data.reset_index(drop = True, inplace = True)
ind.reset_index(drop = True, inplace = True)

# join missing_indicator variables with the rest of the variables
with_mi = cleaned_data.join(ind)

# %% Save data
with_mi.to_csv('X.csv.gzip', compression = 'gzip', index = None)
y.to_csv('y.csv', index = None)
