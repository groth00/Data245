# Data245
ML Project for Data 245 (Predicting Loan Status)

See https://www.kaggle.com/ethon0426/lending-club-20072020q1 for the data.

LendingClub is a P2P lending company where investors can lend money to borrowers. The company has access to personal banking information (credit scores, etc.), but for obvious reasons this dataset excludes those variables. It contains over 100 features (some of which are not applicable), which are listed in the data dictionary from Kaggle.

Our goal is to predict the loan status based on information that is present when the loan application was submitted. This is a binary classification problem, where the labels are fully-paid (0) and charged-off (1). From previous notebooks and preliminary work done here, it is clear that this dataset has in imbalance problem. In other words, the model has overall high accuracy because it has high precision and recall for borrowers that fully paid off their loans, but it cannot detect the borrowers that charged off (low recall).

SMOTE may alleviate the low recall of the model (to be tested).
