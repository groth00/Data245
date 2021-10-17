# Data245
ML Project for Data 245 (Predicting Loan Status)

See https://www.kaggle.com/ethon0426/lending-club-20072020q1 for the data.

LendingClub is a P2P lending company where investors can lend money to borrowers. The company has access to personal banking information (credit scores, etc.), but for obvious reasons this dataset excludes those variables. It contains over 100 features (some of which are not applicable), which are listed in the data dictionary from Kaggle.

Our goal is to predict the loan status based on information that is present when the loan application was submitted. This is a binary classification problem, where the labels are fully-paid (0) and charged-off (1). From previous notebooks and preliminary work done here, it is clear that this dataset has in imbalance problem. In other words, the model has overall high accuracy because it has high precision and recall for borrowers that fully paid off their loans, but it cannot detect the borrowers that charged off (low recall).

SMOTE may alleviate the low recall of the model (to be tested).

Update 1 (9/23): After testing logistic regression and random forest on the full dataset, the recall score was very low (7-8%). The feature importances from RF were used to reduce the amount of features using a threshold of 0.01. I tested Random forest, MLP, and Histogram Gradient Boosting using the reduced feature set and still had around 7% recall. Next, imbalanced-learn was used to under sample the majority class (using RandomUnderSampler). This was applied to individual models (LR, DT, MLP, LinearSVC) and resulted in recall scores of 60%, 57%, 53%, and 63% respectively. I also tested HGB (ensemble) with the under sampler and it achieved a recall of 68%. I also tried BalancedBaggingClassifier (45%), BalancedRandomForestClassifier (64%), and EasyEnsembleClassifier (67%). 

From these results, some of the ensemble methods perform better than the base models. In terms of speed, HGB and BRF were faster than BalancedBagging and EasyEnsemble. The base models were quite fast, finishing within 1-2 minutes.

There are other under sampling techniques available in imbalanced-learn, however, I have only tried RandomUnderSampler. TomekLinks did not finish after a prolonged period of time, and this may occur for the remaining methods, as they are variants on nearest neighbor algorithms.

Next steps: try other sampling methods (under and oversampling), testing more ensemble methods (homogeneous and heterogeneous - VotingClassifier, StackingClassifier)

Update 2 (9/27): I refactored the previous script and added more comments for explainability. The script uses RandomUnderSampler for the majority of testing. 

Individual models (scikit): LR, DT, MLP (best), LinearSVC, SGD

Homogeneous models (scikit): RF, HistGradientBoosting (best)

Homegeneous models (imblearn): BalancedRF, BalancedBagging, EasyEnsemble (best)

Heterogeneous (scikit): VotingClassifier w/majority vote using the individual models above (slightly better than some base models)

Because the size of our dataset is large (>1.8 million rows), the other under and over sampling methods are unable to finish. Computationally, the other methods make use of varying nearest neighbor algorithms, which is too computationally expensive for the amount of features and rows. 

One last note; RandomOverSampler is comparable to RandomUnderSampler in terms of recall for LR, MLP, and SVC. For DT and SGD, the results are worse. In addition, RandomOverSampler takes more significantly more time for all models except for SGD (which is just slightly slower).

Update 3 (10/1): Tested VotingClassifier, StackingClassifier, RandomOverSampler

When using VotingClassifier with a majority vote (hard vote), the same individual models were used as before. When using a soft vote, LinearSVC was substituted for ComplementNB and the loss for SGDClassifier is changed to modified_huber. These are changed because SVM classifiers lack a predict_proba method.

Using uniform weights, VotingClassifier with hard vote outperforms soft vote. Recall improves slightly when adding in ensemble methods.

StackingClassifier (individual + ensemble - SVM) performs better than VotingClassifier at the cost of longer training time.

RandomOverSampler only benefits MLPClassifier slightly and degrades performance on the other 4 individual models.



