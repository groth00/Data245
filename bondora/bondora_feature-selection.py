import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
plt.rcParams['figure.dpi'] = 300

# %%
X = pd.read_csv('Bondora_cleaned.csv')
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
to_plot = sorted_features[sorted_features > 0.01]

ax = plt.subplot()
sns.barplot(x = to_plot.values, y = to_plot.index, ax = ax)
ax.set_title('Feature Importances')

# %%
# use the 20 most important features
best_features = sorted_features[:20]
o = best_features.index

ax = plt.subplot()
sns.barplot(x = best_features.values, y = o, order = o, ax = ax)
ax.set_title('Most Important Features')
ax.set_yticklabels(labels = o, rotation = 0, fontsize = 7)

# %% Save lower-dimensional data
feature_names = best_features.index
X_new = X[feature_names]
X_new.to_csv('BondoraTop20Features.csv', index = None)

