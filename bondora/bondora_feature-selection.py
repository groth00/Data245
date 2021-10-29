import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
plt.rcParams['figure.dpi'] = 300

# %%
X = pd.read_csv('BondoraCleanedUpdated.csv')
y = X['Default'].values.reshape(-1) # 0 is paid, 1 is default
X.drop(['Default'], axis = 1, inplace = True)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.8, random_state = 0)

# %%
rf = RandomForestClassifier(n_jobs = -1, random_state = 0)
rf.fit(X_train, y_train)

# %% Most important features based on RF
importances = {X.columns[i]:val \
               for i,val in enumerate(rf.feature_importances_)}
df_imp = pd.DataFrame.from_dict(importances, 
                                orient = 'index', 
                                columns = ['importance'])

sorted_features = df_imp['importance'].sort_values(ascending = False)
important_vars = sorted_features[sorted_features > 0.01]

# ax = plt.subplot()
# sns.barplot(x = important_vars.values, y = important_vars.index, ax = ax)
# ax.set_title('Feature Importances')

# %%
idx = important_vars.index

ax = plt.subplot()
sns.barplot(x = important_vars.values, y = idx, order = idx, ax = ax)
ax.set_title('Most Important Features')
ax.set_yticklabels(labels = idx, rotation = 0, fontsize = 7)

X_new = X[idx]
# sns.heatmap(X_new.corr(), center = 0)

# %% Save lower-dimensional data
X_new.to_csv('BondoraTopFeatures.csv', index = None)
pd.DataFrame(y, columns='label').to_csv('BondoraY.csv', index = None)


