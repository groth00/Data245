import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
# from sklearn.impute import MissingIndicator
# from sklearn.pipeline import FeatureUnion, make_pipeline

plt.rcParams['figure.dpi'] = 300

# %% 
# Load data
df = pd.read_csv('loan-data-with-missing-vals.csv.gzip', compression = 'gzip')

# %% 
# Check missing values
missing = df.isna().sum().apply(lambda x: x/df.shape[0] * 100)
print(missing[missing > 0].sort_values(ascending = False))

# all_util                 45.506037
# mths_since_recent_inq    12.441885  # consider dropping this?
# employment_len            6.299746  
# bc_util                   3.695852
# percent_bc_gt_75          3.664943
# pct_tl_nvr_dlq            3.638120
# num_accts_ever_120_pd     3.629842
# num_tl_30dpd              3.629842
# num_tl_90g_dpd_24m        3.629842
# pub_rec_bankruptcies      0.037466
# tax_liens                 0.002096

# %% 
# Plot vars with missing values
cols_with_missing_vals = missing[missing > 0]\
    .sort_values(ascending = False).index

for col in cols_with_missing_vals:
    g = sns.histplot(x = df[col])
    plt.show()
    g.clear()

# %% 
# Imputing all_util with mean
i = SimpleImputer(strategy = 'mean')
df['all_util'] = i.fit_transform(df['all_util'].values.reshape(-1, 1))


# %% 
# Impute with IterativeImputer + DecisionTreeRegressor
e = DecisionTreeRegressor(max_features = 'sqrt', 
                          max_depth = 3, random_state = 111)
imp = IterativeImputer(estimator = e)
temp = df.select_dtypes('float64').copy()
imp.fit(temp)
imputed = imp.transform(temp)


# %% 
# Rebuild X, separate y
imputed_num = pd.DataFrame(imputed, 
                           columns = df.dtypes[df.dtypes == 'float64'].index)
cat_vars = df[df.columns[df.dtypes == 'object']]
dummies = pd.get_dummies(cat_vars)
y = imputed_num['y']
imputed_num.drop('y', axis = 1, inplace = True)

X = pd.concat([imputed_num, dummies], axis = 1)

# X.to_csv('X.csv.gzip', compression = 'gzip', index = None)
# y.to_csv('y.csv', index = None)

# %% 
# Scale values + train_test_split
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(X)
scaled_vars = pd.DataFrame(scaler.transform(X), 
                           columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_vars, y, train_size = 0.8, random_state = 111, shuffle = True)

# %% 
# train LogisticRegression
lr = LogisticRegression(solver = 'saga', n_jobs = -1, max_iter = 500)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr.score(X_test, y_test) 

confused = confusion_matrix(y_true = y_test, y_pred = lr_pred)
print(confused)
print(classification_report(y_test, lr_pred))

# 0.807356759284643

# [[295472   3800]
#  [ 67876   4918]]

#               precision    recall  f1-score   support

#          0.0       0.81      0.99      0.89    299272
#          1.0       0.56      0.07      0.12     72794

#     accuracy                           0.81    372066
#    macro avg       0.69      0.53      0.51    372066
# weighted avg       0.76      0.81      0.74    372066

# %%
# train RandomForest
rf = RandomForestClassifier(
    n_estimators = 100,
    criterion = 'gini',
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    bootstrap = True,
    oob_score = True,
    n_jobs = -1,
    random_state = 111)

rf.fit(X_train, y_train.values.reshape(-1))
rf_pred = rf.predict(X_test)

print(rf.score(X_test, y_test))
confused = confusion_matrix(y_true = y_test, y_pred = rf_pred)
print(confused)
print(classification_report(y_test, rf_pred))

# 0.8197067187004456

# [[296043   3229]
#  [ 63852   8942]]

#               precision    recall  f1-score   support

#          0.0       0.82      0.99      0.90    299272
#          1.0       0.73      0.12      0.21     72794

#     accuracy                           0.82    372066
#    macro avg       0.78      0.56      0.55    372066
# weighted avg       0.81      0.82      0.76    372066


# %%
# save model 
dump(rf, 'rf_trained.joblib', compress = 3) 
