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
    recall_score, matthews_corrcoef, roc_auc_score, plot_roc_curve, \
    plot_confusion_matrix, plot_precision_recall_curve

from helper_functions import fitEstimatorWithoutSampling, \
    fitMultipleEstimatorsWithoutSampling, \
    fitEstimatorWithoutSamplerWithCV, \
    fitMultipleEstimatorsWithoutSamplerWithCV, \
    scoreBalancedEstimator, \
    scoreMultipleEstimators
    
# %%
X = pd.read_csv('BondoraTop20Features.csv')
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.75, random_state = 0, shuffle = True)

cv = StratifiedShuffleSplit(n_splits = 10, train_size = 0.8, random_state = 0)
fitted_scikit = fitMultipleEstimatorsWithoutSamplerWithCV(
    X_train, y_train, all_estimators, cv = cv)

# it's not using the updated file for some reason?
pred_scikit = scoreMultipleEstimators(X_test, y_test, fitted_scikit)


# Results for logisticregression 

#               precision    recall  f1-score   support

#          0.0       0.97      0.99      0.98      7912
#          1.0       0.99      0.98      0.99     10034

#     accuracy                           0.99     17946
#    macro avg       0.98      0.99      0.99     17946
# weighted avg       0.99      0.99      0.99     17946
 

# ROC_AUC_score 0.9860829474024918 

# Matthew Correlation Coefficient 0.9703598162921799 

# Results for decisiontreeclassifier 

#               precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00      7912
#          1.0       1.00      1.00      1.00     10034

#     accuracy                           1.00     17946
#    macro avg       1.00      1.00      1.00     17946
# weighted avg       1.00      1.00      1.00     17946
 

# ROC_AUC_score 1.0 

# Matthew Correlation Coefficient 1.0 

# Results for mlpclassifier 

#               precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00      7912
#          1.0       1.00      1.00      1.00     10034

#     accuracy                           1.00     17946
#    macro avg       1.00      1.00      1.00     17946
# weighted avg       1.00      1.00      1.00     17946
 

# ROC_AUC_score 0.9981295395453235 

# Matthew Correlation Coefficient 0.9958268125945396 

# Results for linearsvc 

#               precision    recall  f1-score   support

#          0.0       0.98      1.00      0.99      7912
#          1.0       1.00      0.99      0.99     10034

#     accuracy                           0.99     17946
#    macro avg       0.99      0.99      0.99     17946
# weighted avg       0.99      0.99      0.99     17946
 

# ROC_AUC_score 0.9923455650182705 

# Matthew Correlation Coefficient 0.9829747845874907 

# Results for sgdclassifier 

#               precision    recall  f1-score   support

#          0.0       0.98      0.99      0.98      7912
#          1.0       0.99      0.98      0.99     10034

#     accuracy                           0.99     17946
#    macro avg       0.99      0.99      0.99     17946
# weighted avg       0.99      0.99      0.99     17946
 

# ROC_AUC_score 0.9862836049040946 

# Matthew Correlation Coefficient 0.9714845618242587 

# Results for histgradientboostingclassifier 

#               precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00      7912
#          1.0       1.00      1.00      1.00     10034

#     accuracy                           1.00     17946
#    macro avg       1.00      1.00      1.00     17946
# weighted avg       1.00      1.00      1.00     17946
 

# ROC_AUC_score 1.0 

# Matthew Correlation Coefficient 1.0 

# Results for randomforestclassifier 

#               precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00      7912
#          1.0       1.00      1.00      1.00     10034

#     accuracy                           1.00     17946
#    macro avg       1.00      1.00      1.00     17946
# weighted avg       1.00      1.00      1.00     17946
 

# ROC_AUC_score 1.0 

# Matthew Correlation Coefficient 1.0 

