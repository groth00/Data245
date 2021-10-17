import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import dump
from time import time

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, \
    HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, \
    recall_score

from sklearn.model_selection import KFold, cross_validate

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier, \
    BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline as imblearn_pipeline
from imblearn.metrics import classification_report_imbalanced

plt.rcParams['figure.dpi'] = 300

# %%
#Load data
X = pd.read_csv('X.csv.gzip', compression = 'gzip')
y = pd.read_csv('y.csv')
y = y.values.reshape(-1)

# %%
# helper functions
# scale data before fitting a single model
def fitEstimator(estimator, X_train, y_train, pipeline = False):
    if pipeline:
        p = make_pipeline(MinMaxScaler(), estimator)
    else:
        p = estimator
    start = time()
    print('Started fitting model...')
    p.fit(X_train, y_train)
    end = time()
    print('Finished training model.')
    print(f'It took {end - start} seconds to train the model.')
    return p

# score a single fitted model
def scoreEstimator(trained_estimator, X_test, y_test):
    print(trained_estimator.score(X_test, y_test))
    preds = trained_estimator.predict(X_test)
    # print confusion matrix and plot a heatmap
    confused = confusion_matrix(y_true = y_test, y_pred = preds)
    print(confused)
    ax = plt.subplot()
    # row0: class 0 recall, col0: class 0 precision (same for row1, col1)
    sns.heatmap(data = confused, fmt = ',', annot = True, ax = ax)
    ax.set_title('')
    print(classification_report(y_true = y_test, y_pred = preds))
    return None
    
# %%
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.75, random_state = 111, shuffle = True)

# %%
# LogisticRegression (full features)
lr = fitEstimator(
    LogisticRegression(solver = 'saga', n_jobs = -1, max_iter = 500),
    X_train, y_train)
scoreEstimator(lr, X_test, y_test)

# %%
# RandomForest (full features)
rf = fitEstimator(
    RandomForestClassifier(
        n_estimators = 100, max_depth = None, bootstrap = True,
        oob_score = True, n_jobs = -1, random_state = 111),
    X_train, y_train)
scoreEstimator(rf, X_test, y_test)

# [[293692   3389]
#  [ 67749   4231]]

#               precision    recall  f1-score   support

#          0.0       0.81      0.99      0.89    297081
#          1.0       0.56      0.06      0.11     71980

#     accuracy                           0.81    369061
#    macro avg       0.68      0.52      0.50    369061
# weighted avg       0.76      0.81      0.74    369061

# %%
# Most important features based on RF
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

# let's just use the features that have importance > 0.01 (32 variables)
best_features = sorted_features[sorted_features >= 0.01]
o = best_features.index

ax = plt.subplot()
sns.barplot(x = best_features.values, y = o, order = o, ax = ax)
ax.set_title('Most Important Features')
ax.set_yticklabels(labels = o, rotation = 0, fontsize = 7)

# %%
# Save lower-dimensional data
feature_names = best_features.index
X_new = X[feature_names]
# X_new.to_csv('RF_important_features.csv.gzip', compression = 'gzip')

# %%
# Save a model using joblib
# dump(rf, 'rf_trained.joblib', compress = 3) 

# %%
# Next steps
# try reducing # of features using other feature selection methods

# different methods: individual models, homogeneous ens, heterogeneous ens
# individual: LinearSVC, MLP, LR, DT
# homogeneous ensemble: GBDT, RF
# heterogeneous ensemble: VotingClassifier, StackingClassifier

# based on some testing, the individual models have similar recall (0.07-0.08)
# need to use imbalanced-learn/smote-variants w/models

# cross-validation, tune models (see GridSearch)
# ex. VotingClassifier + GS, StackingClassifier + cv

# finally, follow the approaches in the research papers 
# ex. SOM for consensus, clustering + alone, cluster + alone + consensus

# %%
# testing a few models + sampling methods on reduced data
X = pd.read_csv('RF_important_features.csv.gzip', compression = 'gzip')
y = pd.read_csv('y.csv')
y = y.values.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.75, random_state = 111, shuffle = True)

# %%
# HistGradientBoostingClassifier (fast, <1 min)
p = fitEstimator(HistGradientBoostingClassifier(random_state = 0), X, y)
scoreEstimator(p, X_test, y_test)

# %%
# ANN (~9 minutes)
mlp_model = fitEstimator(MLPClassifier(learning_rate = 'invscaling',
                                       random_state = 0,
                                       early_stopping = True), X, y)
scoreEstimator(mlp_model, X_test, y_test)
# both GB, ANN have similar results to LR, RF -> try imbalanced-learn

# %%
# imbalanced-learn 
# helper f for testing out samplers and estimators
def fitEstimatorWithSamplerWithCV(X, y, cv = None, 
        sampler = RandomUnderSampler(random_state = 0),
        estimator = HistGradientBoostingClassifier(random_state = 0)):
    
    model = imblearn_pipeline(sampler, estimator)
    if cv is None:
        cv = KFold(n_splits = 3, shuffle = True, random_state = 100)
    results = cross_validate(model, X, y, cv = cv,
                             return_train_score = True,
                             return_estimator = True,
                             verbose = 1,
                             scoring = 'recall',
                             n_jobs = -1)
    print('Average recall and standard deviation \n' +
          f"{results['test_score'].mean()} +/- {results['test_score'].std()}")
    bestEstimator = results['estimator'][np.argmax(results['test_score'])]
    return bestEstimator

def fitEstimatorWithSamplerWithoutCV(X_train, y_train, 
        sampler = RandomUnderSampler(random_state = 0),
        estimator = HistGradientBoostingClassifier(random_state = 0)):
    
    model = imblearn_pipeline(sampler, estimator)
    model.fit(X_train, y_train)
    return model

def scoreBalancedEstimator(estimator, X_test, y_test):
    pred = estimator.predict(X_test)
    confused = confusion_matrix(y_true = y_test, y_pred = pred)
    print(confused)
    ax = plt.subplot()
    sns.heatmap(data = confused, fmt = ',', annot = True, ax = ax)
    # print(classification_report(y_true = y_test, y_pred = cv_pred))
    print(classification_report_imbalanced(y_true = y_test, y_pred = pred))
    return None

# %%
# simplest sampler is RandomUnderSampler; TomekLinks takes too long
kfold = KFold(n_splits = 3, shuffle = True, random_state = 100)
est = fitEstimatorWithSamplerWithCV(X = X, y = y, cv = kfold)
scoreEstimator(est, X_test, y_test)

# [[239923 131330]
#  [ 28751  61322]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.89      0.65      0.68      0.75      0.66      0.44    371253
#         1.0       0.32      0.68      0.65      0.43      0.66      0.44     90073

# avg / total       0.78      0.65      0.67      0.69      0.66      0.44    461326

# %%
# BalancedBaggingClassifier, also trying different n_estimators
# recall doesn't improve much from 10 to 30 estimators (also takes more time)
scores = []
estimators = []
for n in range(10, 60, 10):
    bag = fitEstimator(BalancedBaggingClassifier(n_estimators = n,
                                                 n_jobs = -1,
                                                 random_state = 100),
                       X_train, y_train)
    pred = bag.predict(X_test)
    scores.append(recall_score(y_true = y_test, y_pred = pred))
    estimators.append(bag)
    print(classification_report(y_test, pred))
# [[293771  77482]
#  [ 49144  40929]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.86      0.79      0.45      0.82      0.60      0.37    371253
#         1.0       0.35      0.45      0.79      0.39      0.60      0.35     90073

# avg / total       0.76      0.73      0.52      0.74      0.60      0.37    461326


# %%
# BalancedRandomForestClassifier is slightly worse than HGBC, but is fast
# tried with 100 estimators, didn't make a difference
brf = fitEstimator(BalancedRandomForestClassifier(n_estimators = 50,
                                                  n_jobs = -1,
                                                  random_state = 100,
                                                  max_samples = 0.5),
                   X_train, y_train)
scoreEstimator(brf, X_test, y_test)
# 0.6520898453588135
# [[242935 128318]
#  [ 32182  57891]]
#               precision    recall  f1-score   support

#          0.0       0.88      0.65      0.75    371253
#          1.0       0.31      0.64      0.42     90073

#     accuracy                           0.65    461326
#    macro avg       0.60      0.65      0.59    461326
# weighted avg       0.77      0.65      0.69    461326

# %%
# EasyEnsembleClassifier; took 8.3 minutes, worse than BalancedRF
ee = fitEstimator(EasyEnsembleClassifier(n_estimators = 10,
                                          n_jobs = -1,
                                          random_state = 100),
                   X_train, y_train)
scoreEstimator(ee, X_test, y_test)
# It took 492.42835903167725 seconds to train the model.
# 0.6411431395585768
# [[235479 135774]
#  [ 29776  60297]]
#               precision    recall  f1-score   support

#          0.0       0.89      0.63      0.74    371253
#          1.0       0.31      0.67      0.42     90073

#     accuracy                           0.64    461326
#    macro avg       0.60      0.65      0.58    461326
# weighted avg       0.77      0.64      0.68    461326

# %%
# individual models with RandomUnderSampler
# LR i tried 500,1000,2000 iterations, it never converged
# might as well stick with the default (100) to save time
estimator_list = {
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
                     max_iter = 1000)
    }

fit_estimators = []
for name, estimator in estimator_list.items():
    print(f'Started training {name}.')
    start = time()
    est = fitEstimatorWithSamplerWithoutCV(
        X_train, y_train, estimator = estimator)
    end = time()
    print(f'Finished training {name} in {end-start} seconds. \n')
    fit_estimators.append(est)

# Finished training LR in 806.5017838478088 seconds. 

# Finished training DT in 19.58332395553589 seconds. 

# Finished training ANN in 78.81134724617004 seconds. 

# Finished training SVC in 20.717212915420532 seconds.

# %%
# LinearSVC performs the best
for e in fit_estimators:
    scoreBalancedEstimator(e, X_test, y_test)
    
# [[214504 156749]
#  [ 35684  54389]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.86      0.58      0.60      0.69      0.59      0.35    371253
#         1.0       0.26      0.60      0.58      0.36      0.59      0.35     90073

# avg / total       0.74      0.58      0.60      0.63      0.59      0.35    461326

# [[211599 159654]
#  [ 38973  51100]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.84      0.57      0.57      0.68      0.57      0.32    371253
#         1.0       0.24      0.57      0.57      0.34      0.57      0.32     90073

# avg / total       0.73      0.57      0.57      0.61      0.57      0.32    461326

# [[254099 117154]
#  [ 42728  47345]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.86      0.68      0.53      0.76      0.60      0.37    371253
#         1.0       0.29      0.53      0.68      0.37      0.60      0.35     90073

# avg / total       0.75      0.65      0.56      0.68      0.60      0.36    461326

# [[244589 126664]
#  [ 32902  57171]]
#                    pre       rec       spe        f1       geo       iba       sup

#         0.0       0.88      0.66      0.63      0.75      0.65      0.42    371253
#         1.0       0.31      0.63      0.66      0.42      0.65      0.42     90073

# avg / total       0.77      0.65      0.64      0.69      0.65      0.42    461326
