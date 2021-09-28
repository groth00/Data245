import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time

from sklearn.model_selection import train_test_split, KFold, cross_validate, \
    StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, \
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, \
    recall_score

from imblearn.under_sampling import RandomUnderSampler, TomekLinks, \
    EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedBaggingClassifier, \
    BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline as imblearn_pipeline
from imblearn.metrics import classification_report_imbalanced

plt.rcParams['figure.dpi'] = 300
# %% helper functions
# WITHOUT SAMPLING
def fitEstimatorWithoutSampling(X_train, y_train, estimator):
    p = imblearn_pipeline(MinMaxScaler(), estimator)
    start = time()
    print('Started fitting model...')
    p.fit(X_train, y_train)
    end = time()
    print(f'Finished training in {end - start} seconds.')
    return p

def fitMultipleEstimatorsWithoutSampling(X_train, y_train, estimator_dict):
    fit_estimators = []
    for name, estimator in estimator_dict.items():
        fit_estimator = fitEstimatorWithoutSampling(
            X_train, y_train, estimator = estimator)
        fit_estimators.append(fit_estimator)
    return fit_estimators

def scoreEstimator(trained_estimator, X_test, y_test):
    predictions = trained_estimator.predict(X_test)
    print(recall_score(y_test, predictions))

# %% imbalanced-learn functions
def fitEstimatorWithSamplerWithCV(X, y, cv = None, 
        sampler = RandomUnderSampler(random_state = 111),
        estimator = HistGradientBoostingClassifier(random_state = 111)):
    
    model = imblearn_pipeline(MinMaxScaler(), sampler, estimator)
    if cv is None:
        cv = StratifiedShuffleSplit(train_size = 0.25,
                                    random_state = 111)
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

def fitMultipleEstimatorsWithCV(X, y, estimator_dict,
    sampler = RandomUnderSampler(random_state = 111),
    cv = StratifiedShuffleSplit(n_splits = 4, train_size = 0.25,
                                random_state = 111)):
    fit_estimators = []
    for name, estimator in estimator_dict.items():
        print(f'Started training {name} with {sampler}.')
        start = time()
        fit_estimator = fitEstimatorWithSamplerWithCV(
            X, y, cv = cv, sampler = sampler, estimator = estimator)
        end = time()
        print(f'Finished training {name} in {end-start} seconds. \n')
        fit_estimators.append(fit_estimator)
    return fit_estimators

def fitEstimatorWithSamplerWithoutCV(X_train, y_train, 
        sampler = RandomUnderSampler(random_state = 111),
        estimator = HistGradientBoostingClassifier(random_state = 111)):
    ''' 
    default: use RandomUnderSampler, HistGradientBoostingClassifier
    scale features -> under sample majority class -> fit estimator
    '''
    model = imblearn_pipeline(MinMaxScaler(), sampler, estimator)
    model.fit(X_train, y_train)
    return model

def fitMultipleEstimators(X_train, y_train, estimator_dict, 
                          sampling_method = None):
    if not sampling_method:
        sampling_method = RandomUnderSampler(random_state = 111)
    fit_estimators = []
    for name, estimator in estimator_dict.items():
        print(f'Started training {name} with {sampling_method}.')
        start = time()
        fit_estimator = fitEstimatorWithSamplerWithoutCV(
            X_train, y_train, sampler = sampling_method, estimator = estimator)
        end = time()
        print(f'Finished training {name} in {end-start} seconds. \n')
        fit_estimators.append(fit_estimator)
    return fit_estimators

def scoreBalancedEstimator(estimator, X_test, y_test, verbose = 0):
    '''
    compute and return predictions
    move metrics/plotting to a separate function
    '''
    name = estimator.steps[-1][0]
    pred = estimator.predict(X_test)
    if verbose == 0:
        print(f'Recall for {name}: \n',
              recall_score(y_true = y_test, y_pred = pred), '\n')
    else:
        print(classification_report(y_true = y_test, y_pred = pred), '\n')
    return pred

def scoreMultipleEstimators(X_test, y_test, fitted_estimators):
    ''' 
    for each estimator, print the recall and store its predictions
    return predictions for later use (other metrics)
    '''
    list_of_prediction_arrays = []
    for estimator in fitted_estimators:
        predicted = scoreBalancedEstimator(estimator, X_test, y_test)
        list_of_prediction_arrays.append(predicted)
    return list_of_prediction_arrays

def showConfusionMatrixWithHeatmap(y_test, predictions, name = None):
    confused = confusion_matrix(y_true = y_test, y_pred = predictions)
    print(confused, '\n')
    ax = plt.subplot()
    sns.heatmap(data = confused, fmt = ',', annot = True, ax = ax)
    plt.show()
    ax.set_title(f'Confusion Matrix for {name}')
    ax.clear()
    return None

def printClassificationReport(y_test, predictions):
    print(classification_report(y_true = y_test, y_pred = predictions))

def printImbalancedClassificationReport(y_test, predictions):
    print(classification_report_imbalanced(y_test, predictions))


# %% 
X = pd.read_csv('RF_important_features.csv.gzip', compression = 'gzip')
y = pd.read_csv('y.csv')
y = y.values.reshape(-1)


# %% a one time 75/25 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.75, random_state = 111, shuffle = True)

# %% testing individual models
estimator_dict = {
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
    'SGDC': SGDClassifier(max_iter = 1000, verbose = 0,
                          n_jobs = -1, random_state = 111,
                          learning_rate = 'optimal',
                          early_stopping = True,
                          )
    }

fitted = fitMultipleEstimators(X_train, y_train, estimator_dict)

# random state = 111 for estimator and sampler

# Finished training LR in 16.586179971694946 seconds.
# Finished training DT in 17.770429849624634 seconds. 
# Finished training ANN in 180.04376673698425 seconds.
# Finished training SVC in 6.3358142375946045 seconds.
# Finished training SGDC in 3.45185923576355 seconds.


# %% score individual estimators
individual_predictions = scoreMultipleEstimators(y_test, fitted)

# Recall for logisticregression: 
#  0.6314433848100984 
 
# Recall for decisiontreeclassifier: 
#  0.5609228070564987
 
# Recall for mlpclassifier: 
#  0.6917611270858082
 
# Recall for linearsvc: 
#  0.6280461403528249
 
# Recall for sgdclassifier: 
#  0.6448769331542193

# %% ensemble methods from sklearn
ensembles = {
    'HGBC': HistGradientBoostingClassifier(random_state = 111),
    'RF': RandomForestClassifier(
        n_estimators = 100, max_depth = None, bootstrap = True,
        oob_score = True, n_jobs = -1, random_state = 111)
    }

fitted_ensembles = fitMultipleEstimators(X_train, y_train, ensembles)

# random state = 111 for estimator and sampler

# Finished training HGBC in 14.294695377349854 seconds.
# Finished training RF in 169.3908851146698 seconds.


# %%
ensemble_predictions = scoreMultipleEstimators(y_test, fitted_ensembles)

# Recall for histgradientboostingclassifier: 
#  0.6760516469974354

# Recall for randomforestclassifier: 
#  0.6551907896928048 

# %% ensemble methods from imbalanced-learn
imbalanced_ensembles = {
    'BalancedBagging': BalancedBaggingClassifier(n_jobs = -1, 
                                                 random_state = 111),
    'BalancedRF': BalancedRandomForestClassifier(n_jobs = -1,
                                                 random_state = 111),
    'EasyEnsemble': EasyEnsembleClassifier(n_jobs = -1,
                                           random_state = 111),
    }

fitted_imbalanced = fitMultipleEstimatorsWithoutSampling(
    X_train, y_train, imbalanced_ensembles)

# random state = 111 for estimators
# Finished training BalancedBagging in 103.74243307113647 seconds.
# Finished training BalancedRF in 152.8362331390381 seconds.
# Finished training EasyEnsemble in 490.34461307525635 seconds.

# %%
imbalanced_ensemble_predictions = scoreMultipleEstimators(
    y_test, fitted_imbalanced)

# Recall for balancedbaggingclassifier: 
#  0.40160758495886667 

# Recall for balancedrandomforestclassifier: 
#  0.6577331719827251 

# Recall for easyensembleclassifier: 
#  0.6693237707193055



# %% new: VotingClassifier and StackedClassifier
individual_pipelines = [imblearn_pipeline(MinMaxScaler(), 
                   RandomUnderSampler(random_state = 111),
                   estimator) for estimator in estimator_dict.values()]

# majority vote (hard)
# uniform weights for each classifier
v = VotingClassifier(estimators = list(zip(
    [k for k in estimator_dict.keys()], individual_pipelines)),
                     voting = 'hard',
                     weights = None,
                     n_jobs = -1,
                     verbose = True)

v.fit(X_train, y_train)
voter_predictions = v.predict(X_test)

# better than DT, SVC, slightly better than LR, but worse than SGDC and MLP
print(recall_score(y_test, voter_predictions))
# 0.6375606452544048

# ---------------------------------------------------------------------
# %% trying a train_size of 0.25 + StratifiedShuffleSplit

ind_cv = fitMultipleEstimatorsWithCV(
    X, y, estimator_dict, 
    sampler = RandomUnderSampler(random_state = 111),
    cv = StratifiedShuffleSplit(n_splits = 4, 
                                train_size = 0.25, 
                                random_state = 111))

# Started training LR with RandomUnderSampler(random_state=111).
# Average recall and standard deviation 
# 0.6348872801027452 +/- 0.0005213785620505781
# Finished training LR in 20.357626914978027 seconds. 

# Started training DT with RandomUnderSampler(random_state=111).
# Average recall and standard deviation 
# 0.5642379424316111 +/- 0.000916049529759271
# Finished training DT in 20.305978298187256 seconds. 

# Started training ANN with RandomUnderSampler(random_state=111).
# Average recall and standard deviation 
# 0.6691455420955427 +/- 0.011965077509234824
# Finished training ANN in 151.715106010437 seconds. 

# Started training SVC with RandomUnderSampler(random_state=111).
# Average recall and standard deviation 
# 0.6315913658520336 +/- 0.0005496453884532532
# Finished training SVC in 15.03044319152832 seconds. 

# Started training SGDC with RandomUnderSampler(random_state=111).
# Average recall and standard deviation 
# 0.6759511146149092 +/- 0.03589493448214965
# Finished training SGDC in 13.092419147491455 seconds.

# ---------------------------------------------------------------------
# %% other sampling techniques
# tried TomekLinks, ENN with 0.25/0.75, didn't finish
# tried with 0.1/0.9, didn't finish either
# works quickly with 0.01/0.99, but it's pointless 

# RandomUnderSampler > RandomOverSampler in terms of recall and time
ind_os_cv = fitMultipleEstimatorsWithCV(
    X, y, estimator_dict,
    sampler = RandomOverSampler(random_state = 111),
    cv = StratifiedShuffleSplit(n_splits = 4, 
                                train_size = 0.25, 
                                random_state = 111))

# Started training LR with RandomOverSampler(random_state=111).
# Average recall and standard deviation 
# 0.6351935539985861 +/- 0.0005493345411437624
# Finished training LR in 48.57807731628418 seconds. 

# Started training DT with RandomOverSampler(random_state=111).
# Average recall and standard deviation 
# 0.26963761598620195 +/- 0.0017555558910426583
# Finished training DT in 52.16799807548523 seconds. 

# Started training ANN with RandomOverSampler(random_state=111).
# Average recall and standard deviation 
# 0.6761731863218632 +/- 0.015141963654061948
# Finished training ANN in 1103.6012589931488 seconds. 

# Started training SVC with RandomOverSampler(random_state=111).
# Average recall and standard deviation 
# 0.6316644644555727 +/- 0.0005961509814442347
# Finished training SVC in 32.7990562915802 seconds. 

# Started training SGDC with RandomOverSampler(random_state=111).
# Average recall and standard deviation 
# 0.5906857574310744 +/- 0.031092631065272963
# Finished training SGDC in 14.984627962112427 seconds. 