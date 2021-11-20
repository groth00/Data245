import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time

from sklearn.model_selection import cross_validate, \
    StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, \
    recall_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as imblearn_pipeline
from imblearn.metrics import classification_report_imbalanced

# Fit without sampler
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


def fitEstimatorWithoutSamplerWithCV(X, y, estimator, cv = None):
    model = make_pipeline(MinMaxScaler(), estimator)
    if cv is None:
        cv = StratifiedShuffleSplit(n_splits = 4,
                                    train_size = 0.75,
                                    random_state = 111)
    results = cross_validate(model, X, y, cv = cv,
                             return_train_score = False,
                             return_estimator = True,
                             verbose = 1,
                             scoring = 'recall',
                             n_jobs = -1)
    # print('Average recall and standard deviation \n' +
    #       f"{results['test_score'].mean()} +/- {results['test_score'].std()}")
    bestEstimator = results['estimator'][np.argmax(results['test_score'])]
    return bestEstimator

def fitMultipleEstimatorsWithoutSamplerWithCV(X, y, estimator_dict,
    cv = StratifiedShuffleSplit(n_splits = 4, train_size = 0.75,
                                random_state = 111)):
    fit_estimators = []
    for name, estimator in estimator_dict.items():
        print(f'Started training {name}.')
        start = time()
        fit_estimator = fitEstimatorWithoutSamplerWithCV(
            X, y, cv = cv, estimator = estimator)
        end = time()
        print(f'Finished training {name} in {end-start} seconds. \n')
        fit_estimators.append(fit_estimator)
    return fit_estimators

# ----------------------------------------------------------------------------
# Fit with sampler
def fitEstimatorWithSamplerWithCV(X, y, cv = None, 
        sampler = RandomUnderSampler(random_state = 111),
        estimator = HistGradientBoostingClassifier(random_state = 111)):
    
    model = imblearn_pipeline(MinMaxScaler(), sampler, estimator)
    if cv is None:
        cv = StratifiedShuffleSplit(train_size = 0.75,
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
    cv = StratifiedShuffleSplit(n_splits = 4, train_size = 0.75,
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

# ----------------------------------------------------------------------------
def scoreBalancedEstimator(estimator, X_test, y_test):
    '''
    assumes that the estimator param is a pipeline
    the last step of the pipeline is the classifier, hence name = steps[-1][0]
    compute recall score and return predictions
    '''
    name = estimator.steps[-1][0]
    pred = estimator.predict(X_test)
    print(f'Results for {name}')
    print(classification_report(y_true = y_test, y_pred = pred))
    
    try:
        target = estimator.predict_proba(X_test)[:, 1]
    except AttributeError:
        target = estimator.decision_function(X_test)
        
    print('ROC_AUC_score:', roc_auc_score(y_test, 
                                          target))
    print("Matthew's corrcoef:", 
          matthews_corrcoef(y_test, pred))
    print("Cohen's Kappa:", cohen_kappa_score(y_test, pred))
    return pred

def scoreMultipleEstimators(fitted_estimators, X_test, y_test):
    ''' 
    for each estimator, call scoreBalancedEstimator
    return predictions for later use (other metrics)
    '''
    list_of_prediction_arrays = []
    for estimator in fitted_estimators:
        predicted = scoreBalancedEstimator(estimator, X_test, y_test)
        list_of_prediction_arrays.append(predicted)
    return None

def evaluateEstimators(estimators, X_test, y_test):
    for e in estimators:
        evaluateEstimators(e)

def evaluateEstimator(estimator, X_test, y_test):
    pred = estimator.predict(X_test)
    print(f'Results for {estimator.__class__.__name__}')
    print(classification_report(y_true = y_test, y_pred = pred))
    try:
        target = estimator.predict_proba(X_test)[:, 1]
    except AttributeError:
        target = estimator.decision_function(X_test)

    print('ROC_AUC_score:', roc_auc_score(y_test, target))
    print("Matthew's corrcoef:", 
          matthews_corrcoef(y_test, pred))
    print("Cohen's Kappa:", cohen_kappa_score(y_test, pred))
    return None

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

# ----------------------------------------------------------------------------
# for VotingClassifier

def buildPipeline(estimator):
    pipeline = imblearn_pipeline(
        MinMaxScaler(), RandomUnderSampler(random_state = 111), estimator)
    return pipeline

def buildPipelines(estimator_dict,
                   sampler = RandomUnderSampler(random_state = 111)):
    '''
    takes in dictionary
    k = name of estimator
    v = estimator object
    
    returns list of imbalanced-learn pipelines w/MinMaxScaler, RUS, estimator
    
    you can pass this directly into a VotingClassifier
    '''
    pipelines = [
        imblearn_pipeline(MinMaxScaler(), sampler, estimator) \
            for estimator in estimator_dict.values()
        ]
    return list(zip(estimator_dict.keys(), pipelines))

def fitVotingClassifier(voting_classifier, X_train, y_train):
    print('Started training.\n')
    start = time()
    voting_classifier.fit(X_train, y_train)
    end = time()
    print(f'{end - start} seconds elapsed.\n')
    return voting_classifier

def scoreVotingClassifier(voting_classifier, X_test, y_test):
    predictions = voting_classifier.predict(X_test)
    print(f'Recall is {recall_score(y_test, predictions)}')
    return None

# ----------------------------------------------------------------------------
# HalvingGridSearchCV
















