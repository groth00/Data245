import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.pipeline import FeatureUnion, make_pipeline

plt.rcParams['figure.dpi'] = 300

# %%
# Load data
X = pd.read_csv('X.csv.gzip', compression = 'gzip')
y = pd.read_csv('y.csv')
y = y.values.reshape(-1)

# %%
# Scale values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(X)
scaled_vars = pd.DataFrame(scaler.transform(X), columns = X.columns)

# %%
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_vars, y, train_size = 0.8, random_state = 111, shuffle = True)

# %%
# LogisticRegression (baseline w/all features)
lr = LogisticRegression(solver = 'saga', n_jobs = -1, max_iter = 500)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))

confused = confusion_matrix(y_true = y_test, y_pred = lr_pred)
print(confused)

ax = plt.subplot()
sns.heatmap(confused, robust = True, annot = confused, fmt = ',',
            ax = ax)
ax.set_title('Confusion Matrix for LR')

print(classification_report(y_test, lr_pred))

# 0.8079179322659398

# [[293111   3970]
#  [ 66920   5060]]

#               precision    recall  f1-score   support

#          0.0       0.81      0.99      0.89    297081
#          1.0       0.56      0.07      0.12     71980

#     accuracy                           0.81    369061
#    macro avg       0.69      0.53      0.51    369061
# weighted avg       0.76      0.81      0.74    369061

# %%
# RandomForest
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

start = time()
print('Started fitting model...')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
end = time()
print('Finished training model.')
print(f'It took {end - start} seconds to train the model.')

print(rf.score(X_test, y_test))
confused = confusion_matrix(y_true = y_test, y_pred = rf_pred)
print(confused)

ax = plt.subplot()
sns.heatmap(confused, robust = True, annot = confused, fmt = ',',
            ax = ax)
ax.set_title('Confusion Matrix for RF')
print(classification_report(y_test, rf_pred))

# 0.8072459566304757

# [[293692   3389]
#  [ 67749   4231]]

#               precision    recall  f1-score   support

#          0.0       0.81      0.99      0.89    297081
#          1.0       0.56      0.06      0.11     71980

#     accuracy                           0.81    369061
#    macro avg       0.68      0.52      0.50    369061
# weighted avg       0.76      0.81      0.74    369061

# %%
# Determining the most important features based on RF
importances = {X.columns[i]:val \
               for i,val in enumerate(rf.feature_importances_)}
df_imp = pd.DataFrame.from_dict(importances, 
                                orient = 'index', 
                                columns = ['importance'])

sorted_features = df_imp['importance'].sort_values(ascending = False)

ax = plt.subplot()
sns.barplot(y = sorted_features.values, x = sorted_features.index, ax = ax)
ax.set_xticks([])
ax.set_title('Feature Importances (for visual purposes)')

# let's just use the features that have importance > 0.01
# 32 variables
best_features = sorted_features[sorted_features >= 0.01]
o = best_features.index

ax = plt.subplot()
sns.barplot(x = best_features.values, y = o, order = o, ax = ax)
ax.set_title('Most Important Features')
ax.set_yticklabels(labels = o, rotation = 0, fontsize = 7)

# %%
# Save
feature_names = best_features.index
X_new = X[feature_names]
# X_new.to_csv('RF_important_features.csv.gzip', compression = 'gzip')

# %%
# Save a model using joblib
# dump(rf, 'rf_trained.joblib', compress = 3) 

# %%
# todo
# use different proportions of train/test data
# reduce features by using feature importance from RF/feature selection methods
# cross-validation
# different estimators
# tune models (see GridSearch)
# SMOTE libraries


