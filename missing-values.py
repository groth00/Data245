import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import time

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

plt.rcParams['figure.dpi'] = 300

# %% Load data
df = pd.read_csv('loan-data-with-missing-vals.csv.gzip', compression = 'gzip')

# %% Check missing values
missing = df.isna().sum().apply(lambda x: x/df.shape[0] * 100)
print(missing[missing > 0].sort_values(ascending = False))

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

# %% Plot vars with missing values
cols_with_missing_vals = missing[missing > 0]\
    .sort_values(ascending = False).index

for col in cols_with_missing_vals:
    g = sns.histplot(x = df[col])
    plt.show()
    g.clear()

# %% Imputing values

# try using IterativeImputer + ExtraTreeRegressor
e = ExtraTreesRegressor(n_estimators = 10, random_state = 111)
i = IterativeImputer(estimator = e)

temp = df.select_dtypes('float64').copy()
start = time()
temp_imputed = i.fit_transform(X = temp)
end = time() - start
print(end)





# %% todo
    
