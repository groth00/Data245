import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import joblib
from time import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, \
    StratifiedShuffleSplit, HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, \
    HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, \
    confusion_matrix

from helper_functions import fitMultipleEstimatorsWithoutSamplerWithCV, \
    scoreMultipleEstimators, evaluateEstimator, evaluateEstimators
    
plt.rcParams['figure.dpi'] = 300
    
# %% read data
# use TopFeatures for 19 variables
# use CleanedUpdated for all 130 variables 
X = pd.read_csv('BondoraTopFeatures.csv')
X_original = pd.read_csv('BondoraCleanedUpdated.csv')
y = pd.read_csv('BondoraY.csv').values.reshape(-1)

# %%
all_estimators = {
    'LR': LogisticRegression(max_iter = 100, 
                             n_jobs = -1, 
                             random_state = 111,
                             solver = 'saga'),
    'DT': DecisionTreeClassifier(random_state = 111),
    'ANN': MLPClassifier(learning_rate = 'invscaling',
                         random_state = 111,
                         early_stopping = True),
    'SVC': LinearSVC(dual = False,
                     random_state = 111,
                     max_iter = 1000),
    'SGDC': SGDClassifier(max_iter = 1000, n_jobs = -1, random_state = 111,
                          learning_rate = 'optimal',
                          early_stopping = True),
    'HGBC': HistGradientBoostingClassifier(random_state = 111),
    'RF': RandomForestClassifier(
        n_estimators = 100, max_depth = None, bootstrap = True,
        oob_score = True, n_jobs = -1, random_state = 111),
    'GBM': lgb.LGBMClassifier(random_state = 0, n_jobs = -1)}

# %%
estimators2 = {
    'HGBC': HistGradientBoostingClassifier(random_state = 0),
    'RF': RandomForestClassifier(
        n_estimators = 100, max_depth = None, bootstrap = True,
        oob_score = True, n_jobs = -1, random_state = 0),}

# %% 
# I stick with all features since the results are slightly better
# than just 19 features
X_train, X_test, y_train, y_test = train_test_split(
    X_original, y, train_size = 0.75, random_state = 0, shuffle = True)

# %%
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
    X, y, train_size = 0.75, random_state = 0, shuffle = True)

# %% Note: I ran this with 19 vars, then ran RF and HGB with all vars
cv = StratifiedShuffleSplit(n_splits = 5, train_size = 0.8, random_state = 0)
fitted_scikit = fitMultipleEstimatorsWithoutSamplerWithCV(
    X_train, y_train, all_estimators, cv = cv)

scoreMultipleEstimators(fitted_scikit, X_test, y_test)

# %%
fitted_scikit2 = fitMultipleEstimatorsWithoutSamplerWithCV(
    X_train_small, y_train_small, all_estimators, cv = cv)

scoreMultipleEstimators(fitted_scikit2, X_test_small, y_test_small)

# %% tune HistGB
hgbc = HistGradientBoostingClassifier(random_state = 0,)
params = {'learning_rate': [0.001, 0.01, 0.1],
          'max_depth': [None, 15, 30],
          'l2_regularization': [0, 0.5, 1.0]}
gs = HalvingGridSearchCV(hgbc, param_grid = params,
                         random_state = 0, n_jobs = -1,
                         verbose = 1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
# {'l2_regularization': 0, 'learning_rate': 0.1, 'max_depth': 15} 
# 0.7428645294725957

# %% HistGB results
# can use the gridsearch predictor directly/train a new model with new params
evaluateEstimator(gs, X_test, y_test, )

#               precision    recall  f1-score   support

#          0.0       0.76      0.53      0.63      7946
#          1.0       0.73      0.89      0.80     11403

#     accuracy                           0.74     19349
#    macro avg       0.75      0.71      0.71     19349
# weighted avg       0.74      0.74      0.73     19349
 
# ROC_AUC_score 0.7081375481553897
# Matthew Correlation Coefficient 0.4535971220074581
# Cohen's Kappa 0.43650131848895246

# %% tune RandomForest
# class_weight = 'balanced_subsample' is slightly worse
rf = RandomForestClassifier(random_state = 0, n_jobs = -1)
rf_params = {'max_depth': [None, 15, 30],
             'max_features': ['sqrt', 'log2'],
             'n_estimators': [100, 150],}

gs_rf = HalvingGridSearchCV(rf, param_grid = rf_params,
                         random_state = 0, n_jobs = -1,
                         verbose = 1, scoring = 'accuracy')
gs_rf.fit(X_train, y_train)
print(gs_rf.best_params_, gs_rf.best_score_)
# for scoring = 'accuracy' (default)
# {'max_depth': 30, 'max_features': 'sqrt', 'n_estimators': 150} 
# 0.7498966230186079

# for scoring = 'f1_weighted'
# {'max_depth': 30, 'max_features': 'sqrt', 'n_estimators': 150} 
# 0.7394069695430456

# %%
evaluateEstimator(gs_rf, X_test, y_test,)

#               precision    recall  f1-score   support

#          0.0       0.78      0.55      0.64      7946
#          1.0       0.74      0.89      0.81     11403

#     accuracy                           0.75     19349
#    macro avg       0.76      0.72      0.73     19349
# weighted avg       0.76      0.75      0.74     19349

# ROC_AUC_score: 0.720529175283157
# Matthew Correlation Coefficient: 0.4792806679725238
# Cohen's Kappa: 0.46201741511192207

# %% tune LightGBM
gbmc = lgb.LGBMClassifier(random_state = 0, n_jobs = -1)
gbmc_params = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 150, 200],
    'reg_alpha': [0, 0.5, 1.0],
    'reg_lambda': [0, 0.5, 1.0]}
gs_gbm = HalvingGridSearchCV(gbmc, param_grid = gbmc_params,
                             random_state = 0, n_jobs = -1,
                             verbose = 1)
gs_gbm.fit(X_train, y_train)
print(gs_gbm.best_params_, gs_gbm.best_score_)
# {'boosting_type': 'gbdt', 'learning_rate': 0.1, 
# 'n_estimators': 100, 'reg_alpha': 1.0, 'reg_lambda': 1.0} 
# 0.7434089145616001

# %% Results from tuned LightGBM
evaluateEstimator(gs_gbm, X_test, y_test)

#               precision    recall  f1-score   support

#          0.0       0.76      0.53      0.63      7946
#          1.0       0.73      0.88      0.80     11403

#     accuracy                           0.74     19349
#    macro avg       0.75      0.71      0.71     19349
# weighted avg       0.74      0.74      0.73     19349

# ROC_AUC_score 0.7082178388680287
# Matthew Correlation Coefficient 0.452742539581331
# Cohen's Kappa 0.4363031428560906

# %% Training models with tuned parameters
hgb_tuned = HistGradientBoostingClassifier(
    l2_regularization = 0, 
    learning_rate = 0.1, 
    max_depth = 15,
    random_state = 0,)
hgb_tuned.fit(X_train, y_train)

rf_tuned = RandomForestClassifier(
    random_state = 0,
    n_jobs = -1,
    max_depth = 30,
    max_features = 'sqrt',
    n_estimators = 200,
    class_weight = {0: 1.2, 1:1})
rf_tuned.fit(X_train, y_train)

gbm = lgb.LGBMClassifier(
    boosting_type = 'gbdt',
    learning_rate = 0.1,
    n_estimators = 100,
    reg_alpha = 1.0,
    reg_lambda = 1.0,
    random_state = 0,
    n_jobs = -1,)
gbm.fit(X_train, y_train)


# %% Plot confusion matrices
fig, axes = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True,
                         figsize = (14, 4))
preds = [hgb_tuned.predict(X_test), rf_tuned.predict(X_test), 
         gbm.predict(X_test)]
names = ['HistGradientBoosting', 'RandomForest', 'LightGBM']

for ax, pred, name in zip(axes.ravel(), preds, names):
    sns.heatmap(
        confusion_matrix(
            y_true = y_test, 
            y_pred = pred),
        annot = True,
        xticklabels = ['Paid', 'Default'],
        yticklabels = ['Paid', 'Default'],
        fmt = ',',
        ax = ax)
    ax.set_title(f'Confusion Matrix for {name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# %% ROC/AUC plot

'''ROC:
plots the true positive rate (TPR) vs. the false positive rate (FPR)
for different probability thresholds

a random classifier would have a curve = to the 45 degree line (diagonal line)
curves closer to the top left corner are better
'''

'''AUC: 
probability that a randomly chosen positive instance is 
ranked higher than a randomly chosen negative instance
'''

hgb_roc = RocCurveDisplay.from_estimator(hgb_tuned, X_test, y_test)
rf_roc = RocCurveDisplay.from_estimator(rf_tuned, X_test, y_test, ax = hgb_roc.ax_)
gbm_roc = RocCurveDisplay.from_estimator(gbm, X_test, y_test, ax = rf_roc.ax_)
rf_roc.ax_.set_title('ROC Curve')
plt.show()

# %% Precision/Recall Curve
'''
Precision = correctly identifying positive out of all positives(TP / TP + FP)
Recall = correctly identifying all samples belonging to a class (TP / TP + FN)
The curve shows the tradeoff between recall and precision for different 
probability thresholds
The best curve would be similar to the ROC curve, but flipped on the y-axis
'''

hgb_pr = PrecisionRecallDisplay.from_estimator(hgb_tuned, X_test, y_test)
rf_pr = PrecisionRecallDisplay.from_estimator(rf_tuned, X_test, y_test, ax = hgb_pr.ax_)
gbm_pr = PrecisionRecallDisplay.from_estimator(gbm, X_test, y_test, ax = rf_pr.ax_)
rf_pr.ax_.set_title('Precision Recall Curve')
plt.show()

# %% RandomForest feature importances
importances = {rf_tuned.feature_names_in_[i]:val \
               for i, val in enumerate(rf_tuned.feature_importances_)}
df_imp = pd.DataFrame.from_dict(importances, 
                                orient = 'index', 
                                columns = ['importance'])
sorted_features = df_imp['importance'].sort_values(ascending = False)
important_vars = sorted_features[:20]

ax = plt.subplot()
sns.barplot(x = important_vars.values, y = important_vars.index, ax = ax)
ax.set_title('Feature Importances')
ax.set_yticklabels(labels = important_vars.index, fontsize = 8.5)

# %% Plot the first decision tree from RandomForest
fig, ax = plt.subplots(1, 1, dpi = 800)
plot_tree(rf_tuned.estimators_[0],
          max_depth = 3,
          feature_names = X_original.columns,
          class_names = ['Paid', 'Default'],
          filled = True,)

# %% GBM feature importances
importances = {gbm.feature_name_[i]:val \
               for i,val in enumerate(gbm.feature_importances_)}
df_imp = pd.DataFrame.from_dict(importances, 
                                orient = 'index', 
                                columns = ['importance'])
sorted_features = df_imp['importance'].sort_values(ascending = False)
important_vars = sorted_features[:20]

ax = plt.subplot()
sns.barplot(x = important_vars.values, y = important_vars.index, ax = ax)
ax.set_title('Feature Importances')
ax.set_yticklabels(labels = important_vars.index, fontsize = 8.5)

# %% Save and load model
joblib.dump(rf_tuned, filename = 'rf_tuned_model', compress = 8)
model = joblib.load('rf_tuned_model')