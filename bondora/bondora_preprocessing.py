import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

plt.rcParams['figure.dpi'] = 300

# %%
df = pd.read_csv('Bondora_dataset/Bondora_preprocessed.csv')

# %% dropping variables and checking what values to fill

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

# county, city have too many categories, LoanDate, year lacks context
df.drop(labels = ['County', 'City', 'LoanDate', 'year'], 
        axis = 1, inplace = True)

# drop redundant variables 
df.drop(labels = ['NoOfPreviousLoansBeforeLoan',
                  'PreviousRepaymentsBeforeLoan',
                  'LanguageCode'],
        axis = 1, inplace = True)

# drop invalid vars (info we wouldn't know at the time the loan is requested)
# also we already have LoanPeriod, no need for other dates
df.drop(labels = ['InterestAndPenaltyPaymentsMade',
                 'InterestAndPenaltyBalance', 'PrincipalBalance', 
                 'PrincipalPaymentsMade',
                 'FirstPaymentDate', 'LastPaymentOn',
                 'MaturityDate_Original', 'MaturityDate_Last'], 
        axis = 1, inplace = True)

df.fillna(value = {'CreditScoreEsMicroL': 'Other',
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

# %% checking correlation between numerical variables
temp = df.copy()
numerical = temp.columns[df.dtypes != 'object']
num_vars = temp[numerical]
sns.heatmap(num_vars.corr())

# Amount is highly correlated with AppliedAmount and MonthlyPayment
# BidsPortfolioManager is highly correlated with AppliedAmount/Amount

# but really, only Amount should be removed because it's redundant
# MonthlyPayment and BidsPortfolioManager could still be useful

df.drop(labels = ['AppliedAmount'],
        axis = 1, inplace = True)

# %%
# (optional) check correlations manually
corrs = num_vars.corr()
correlated_feature_pairs = []
for i, row in enumerate(corrs.values):
    for j, corr_val in enumerate(row):
        if 1 > corr_val > 0.5:
            if (i, j) not in correlated_feature_pairs \
                or (j, i) not in correlated_feature_pairs:
                correlated_feature_pairs.append((i, j))

for pair in correlated_feature_pairs:
    print(f'({num_vars.columns[pair[0]]}, {num_vars.columns[pair[1]]})')

# %%
'''bids (investment offers) manual, # of bids through api,
age, applied amount, amount, interest (max interest rate accepted in the application),
'''
# outliers in age, liabilitiestotal

# kdeplot
v = ['BidsPortfolioManager', 'BidsApi', 'BidsManual',
     'Interest', 'LoanDuration', 'MonthlyPayment',
     'IncomeTotal', 'ExistingLiabilities',
     'LiabilitiesTotal', 'DebtToIncome', 
     'MonthlyPaymentDay', 
     'PreviousEarlyRepaymentsCountBeforeLoan',]

# boxplot
b_vars = ['Age', 'Amount',
          'LiabilitiesTotal']

# %%
to_plot = num_vars.sample(frac = 0.10, random_state = 111)

# %%
for col in v:
    ax = plt.subplot()
    sns.histplot(data = to_plot, x = col, hue = 'Default',
                multiple = 'stack', ax = ax)
    ax.set_title(f'{col}')
    plt.show()
    


# %%
y_to_plot = to_plot['Default'].astype(str)
for col in b_vars:
    ax = plt.subplot()
    sns.boxplot(data = to_plot, x = col, y = y_to_plot,
                ax = ax, order = ['0', '1'])
    ax.set_title(f'{col}')
    ax.set_xlabel('')
    plt.show()
    
# %%
selected = ['BidsPortfolioManager', 'Interest',
            'MonthlyPayment', 'IncomeTotal']
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (9, 9))
for var, ax in zip(selected, axes.ravel()):
    sns.boxplot(data = to_plot, x = var, y = y_to_plot,
                ax = ax, order = ['0', '1'])

    
# %% drop outliers
df.drop(df[df['Age'] < 18].index, axis = 0, inplace = True)
df.drop(df[df['LiabilitiesTotal'] > 30000].index, axis = 0, inplace = True)

# %%
temp = df.copy()
numerical = temp.columns[df.dtypes != 'object']
num_vars = temp[numerical]
new_sample = num_vars.sample(frac = 0.1, random_state = 0)

# %%
y_new = new_sample['Default'].astype(str)
for col in b_vars:
    ax = plt.subplot()
    sns.boxplot(data = new_sample, x = col, y = y_new,
                ax = ax)
    ax.set_title(f'{col}')
    ax.set_xlabel('')
    plt.show()


# %%
missing = df.isna().sum().apply(lambda x: x / df.shape[0])\
    .sort_values(ascending = False)
print(missing[missing > 0])
print(df[missing[missing > 0].index].dtypes)

# only FreeCash and DebtToIncome have a small amount of missing values

# %%
cat_vars = df.columns[df.dtypes == 'object']
dummied = pd.get_dummies(df, columns = cat_vars)

# %% fill Numerical Variables with IterativeImputer 
# (DebtToIncome, FreeCash)
imp = IterativeImputer(
    estimator = DecisionTreeRegressor(max_features = 'log2',
                                      max_depth = None,
                                      random_state = 0))
imp.fit(dummied)
new = pd.DataFrame(imp.transform(dummied), columns = dummied.columns)

# %% save preprocessed data
new.to_csv(path_or_buf = 'BondoraCleanedUpdated.csv', index = None)

# %% OLD ----------- date variables: 
# LoanDate, FirstPaymentDate, LastPaymentOn
# MaturityDate_Original, MaturityDate_Last

# loanPeriodDelta = df['LastPaymentOn'].astype('datetime64') - \
#     df['FirstPaymentDate'].astype('datetime64')
# maturityPeriodDelta = df['MaturityDate_Last'].astype('datetime64') - \
#     df['MaturityDate_Original'].astype('datetime64')

# df['LoanPeriodDays'] = loanPeriodDelta.dt.days
# df['MaturityPeriodDays'] = maturityPeriodDelta.dt.days
# df.drop(labels = ['LastPaymentOn', 'FirstPaymentDate', 'MaturityDate_Last',
#                   'MaturityDate_Original'], axis = 1, inplace = True)
# note: these variables have negative periods

