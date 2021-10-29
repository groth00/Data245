import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time

from sklearn.experimental import enable_hist_gradient_boosting, \
    enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_validate, \
    StratifiedShuffleSplit, HalvingGridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, \
    HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, \
    matthews_corrcoef, roc_auc_score, plot_roc_curve, \
    plot_confusion_matrix, plot_precision_recall_curve

from helper_functions import fitEstimatorWithoutSampling, \
    fitMultipleEstimatorsWithoutSampling, \
    fitEstimatorWithoutSamplerWithCV, \
    fitMultipleEstimatorsWithoutSamplerWithCV, \
    scoreBalancedEstimator, \
    scoreMultipleEstimators
    
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
        oob_score = True, n_jobs = -1, random_state = 111)}

# %%
estimators2 = {
    'HGBC': HistGradientBoostingClassifier(random_state = 0),
    'RF': RandomForestClassifier(
        n_estimators = 100, max_depth = None, bootstrap = True,
        oob_score = True, n_jobs = -1, random_state = 0),
    }

# %% 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.75, random_state = 0, shuffle = True)

cv = StratifiedShuffleSplit(n_splits = 5, train_size = 0.8, random_state = 0)
fitted_scikit = fitMultipleEstimatorsWithoutSamplerWithCV(
    X_train, y_train, all_estimators, cv = cv)

pred_scikit = scoreMultipleEstimators(X_test, y_test, fitted_scikit)

# ----------------------------
# DONE USING 19 VARIABLES ONLY
# ----------------------------

# Results for logisticregression 
#               precision    recall  f1-score   support
#          0.0       0.69      0.52      0.59      7946
#          1.0       0.72      0.84      0.77     11403

#     accuracy                           0.71     19349
#    macro avg       0.70      0.68      0.68     19349
# weighted avg       0.71      0.71      0.70     19349
# ROC_AUC_score 0.6800966927532571 
# Matthew Correlation Coefficient 0.383654882609942 

# Results for decisiontreeclassifier 
#               precision    recall  f1-score   support
#          0.0       0.57      0.57      0.57      7946
#          1.0       0.70      0.70      0.70     11403
         
#     accuracy                           0.65     19349
#    macro avg       0.64      0.64      0.64     19349
# weighted avg       0.65      0.65      0.65     19349
# ROC_AUC_score 0.6357747404822064 
# Matthew Correlation Coefficient 0.27173810833581175 

# Results for mlpclassifier 
#               precision    recall  f1-score   support
#          0.0       0.76      0.47      0.58      7946
#          1.0       0.71      0.89      0.79     11403

#     accuracy                           0.72     19349
#    macro avg       0.73      0.68      0.69     19349
# weighted avg       0.73      0.72      0.70     19349
# ROC_AUC_score 0.6832720331676685 
# Matthew Correlation Coefficient 0.41243318828986597 

# Results for linearsvc 
#               precision    recall  f1-score   support
#          0.0       0.70      0.51      0.59      7946
#          1.0       0.71      0.85      0.77     11403

#     accuracy                           0.71     19349
#    macro avg       0.71      0.68      0.68     19349
# weighted avg       0.71      0.71      0.70     19349
# ROC_AUC_score 0.6795659683835811 
# Matthew Correlation Coefficient 0.3849687551185018 

# Results for sgdclassifier 
#               precision    recall  f1-score   support
#          0.0       0.85      0.27      0.41      7946
#          1.0       0.65      0.97      0.78     11403

#     accuracy                           0.68     19349
#    macro avg       0.75      0.62      0.59     19349
# weighted avg       0.73      0.68      0.63     19349
# ROC_AUC_score 0.6179585127789373 
# Matthew Correlation Coefficient 0.345069874905473 

# Results for histgradientboostingclassifier 
#               precision    recall  f1-score   support

#          0.0       0.76      0.51      0.62      7946
#          1.0       0.72      0.89      0.80     11403
#     accuracy                           0.74     19349
#    macro avg       0.74      0.70      0.71     19349
# weighted avg       0.74      0.74      0.72     19349
# ROC_AUC_score 0.7021821956189016 
# Matthew Correlation Coefficient 0.4448685712061779 

# Results for randomforestclassifier 
#               precision    recall  f1-score   support
#          0.0       0.75      0.54      0.63      7946
#          1.0       0.73      0.87      0.80     11403

#     accuracy                           0.74     19349
#    macro avg       0.74      0.71      0.71     19349
# weighted avg       0.74      0.74      0.73     19349
# ROC_AUC_score 0.7081503450050535 
# Matthew Correlation Coefficient 0.44730356761226053 

# ----------------------------------------------------------------
# very slight improvements when using all features
# ----------------------------------------------------------------

# Results for histgradientboostingclassifier 
#               precision    recall  f1-score   support
#          0.0       0.76      0.53      0.62      7946
#          1.0       0.73      0.89      0.80     11403

#     accuracy                           0.74     19349
#    macro avg       0.75      0.71      0.71     19349
# weighted avg       0.74      0.74      0.73     19349
# ROC_AUC_score 0.7071173704978129 
# Matthew Correlation Coefficient 0.4519420688752592 

# Results for randomforestclassifier 
#               precision    recall  f1-score   support

#          0.0       0.77      0.54      0.64      7946
#          1.0       0.74      0.89      0.80     11403

#     accuracy                           0.75     19349
#    macro avg       0.75      0.72      0.72     19349
# weighted avg       0.75      0.75      0.74     19349
# ROC_AUC_score 0.7150649646227532 
# Matthew Correlation Coefficient 0.46606905401973525 


# %% trying to tune parameters
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

# %%
# will give the same result if using the gridsearch predictor
pred = gs.predict(X_test)

print(classification_report(y_test, pred)) 
print(matthews_corrcoef(y_test, pred))
print(roc_auc_score(y_test, pred))

plot_roc_curve(gs, X_test, y_test)
plot_confusion_matrix(gs, X_test, y_test, values_format = ',')

#               precision    recall  f1-score   support
#          0.0       0.76      0.53      0.63      7946
#          1.0       0.73      0.89      0.80     11403

#     accuracy                           0.74     19349
#    macro avg       0.75      0.71      0.71     19349
# weighted avg       0.74      0.74      0.73     19349

# 0.4535971220074581
# 0.7081375481553897

# %%
# class_weight = 'balanced_subsample' is slightly worse
rf = RandomForestClassifier(random_state = 0, n_jobs = -1)
rf_params = {'max_depth': [None, 15, 30],
             'max_features': ['sqrt', 'log2'],
             'n_estimators': [100, 200],}

gs_rf = HalvingGridSearchCV(rf, param_grid = rf_params,
                         random_state = 0, n_jobs = -1,
                         verbose = 1)
gs_rf.fit(X_train, y_train)
print(gs_rf.best_params_, gs_rf.best_score_)
# {'max_depth': 30, 'max_features': 'sqrt', 'n_estimators': 200} 
# 0.7498966230186079

# %%
rf_pred = gs_rf.predict(X_test)

print(classification_report(y_test, rf_pred)) 
print(matthews_corrcoef(y_test, rf_pred))
print(roc_auc_score(y_test, rf_pred))

plot_roc_curve(gs_rf, X_test, y_test)
plot_confusion_matrix(gs_rf, X_test, y_test, values_format = ',')

#               precision    recall  f1-score   support

#          0.0       0.78      0.54      0.64      7946
#          1.0       0.74      0.90      0.81     11403

#     accuracy                           0.75     19349
#    macro avg       0.76      0.72      0.73     19349
# weighted avg       0.76      0.75      0.74     19349

# 0.4780508317705175
# 0.7193105443679415

