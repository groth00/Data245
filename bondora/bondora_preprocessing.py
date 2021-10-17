import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %%
df = pd.read_csv('Bondora_dataset/Bondora_preprocessed.csv')

# %%
missing = df.isna().sum().apply(lambda x: x / df.shape[0])\
    .sort_values(ascending = False)
print(missing[missing > 0])
print(df[missing[missing > 0].index].dtypes)

# %%
# CreditScoreEsMicroL -> Other
# County -> Other
# LastPaymentOn -> Drop These
# City -> Other
# Rating -> Other
# HomeOwnershipType -> Not_specified
# EmploymentDurationCurrentEmployer -> Other
# EmploymentStatus -> Not_specified
# OccupationArea -> Not_specified
# VerificationType -> Not_set
# Education -> Not_present
# Gender -> Unknown
# MaritalStatus -> Not_specified

# remove rows with missing LastPaymentOn, can't impute this value
df.dropna(subset = ['LastPaymentOn'], axis = 0, inplace = True)

df.fillna(value = {'CreditScoreEsMicroL': 'Other',
                   'County': 'Other',
                   'City': 'Other',
                   'Rating': 'Other',
                   'HomeOwnershipType': 'Not_specified',
                   'EmploymentDurationCurrentEmployer': 'Other',
                   'EmploymentStatus': 'Not_specified',
                   'OccupationArea': 'Not_specified',
                   'VerificationType': 'Not_set',
                   'Education': 'Not_present',
                   'Gender': 'Unknown',
                   'MaritalStatus': 'Not_specified'}, axis = 0, inplace = True)

# %% checking number of categories per categorical variable

cat_vars = df.columns[df.dtypes == 'object']
for i, c in enumerate(cat_vars):
    print(cat_vars[i], len(df[c].unique()))
    
# drop county, city, LoanDate (without context, this has no information)
df.drop(labels = ['County', 'City', 'LoanDate'], axis = 1, inplace = True)
    
# %% deal with date variables: 
# LoanDate, FirstPaymentDate, LastPaymentOn, MaturityDate_Original, MaturityDate_Last

loanPeriodDelta = df['LastPaymentOn'].astype('datetime64') - \
    df['FirstPaymentDate'].astype('datetime64')
maturityPeriodDelta = df['MaturityDate_Last'].astype('datetime64') - \
    df['MaturityDate_Original'].astype('datetime64')

df['LoanPeriodDays'] = loanPeriodDelta.dt.days
df['MaturityPeriodDays'] = maturityPeriodDelta.dt.days
df.drop(labels = ['LastPaymentOn', 'FirstPaymentDate', 'MaturityDate_Last',
                  'MaturityDate_Original'], axis = 1, inplace = True)
# these variables still have negative periods, not sure what to do with it

# %%
cat_vars = df.columns[df.dtypes == 'object']
dummied = pd.get_dummies(df, columns = cat_vars)

# %% fill Numerical Variables with IterativeImputer 
# (PreviousRepaymentsBeforeLoan, MonthlyPayment, DebtToIncome, FreeCash)

# just a warning, did not reach early stopping criterion
imp = IterativeImputer(
    estimator = DecisionTreeRegressor(max_features = 'log2',
                                      max_depth = 8,
                                      random_state = 0))
imp.fit(dummied)
new = pd.DataFrame(imp.transform(dummied), columns = dummied.columns)

# %%
new.to_csv(path_or_buf = 'Bondora_cleaned.csv', index = None)

