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
