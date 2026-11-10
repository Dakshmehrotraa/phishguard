# PhishGuard - results summary

## URL  -  PhiUSIIL (UCI #967)

Rows: 171,307 | Balancing: imblearn.SMOTE | Selected (best validation F1): **Ensemble (LR + NB, soft vote)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Ensemble (LR + NB, soft vote) | 0.8067 | 0.6112 | 0.6297 | 0.6203 | 0.8321 | 0.1341 |
| Multinomial Naive Bayes | 0.8018 | 0.6026 | 0.6158 | 0.6091 | 0.8255 | 0.1359 |
| Logistic Regression | 0.7906 | 0.5728 | 0.6494 | 0.6087 | 0.8253 | 0.1621 |
| Random Forest | 0.8183 | 0.6771 | 0.5265 | 0.5924 | 0.8158 | 0.0840 |

Final model on test set: accuracy 0.8062, precision 0.6083, recall 0.6378, F1 0.6227, ROC-AUC 0.8344, confusion matrix {'tn': 22141, 'fp': 3529, 'fn': 3112, 'tp': 5480}

## TEXT  -  UCI SMS Spam Collection

Rows: 5,574 | Balancing: imblearn.SMOTE | Selected (best validation F1): **Logistic Regression**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9919 | 0.9930 | 0.9463 | 0.9691 | 0.9967 | 0.0010 |
| Ensemble (LR + RF, soft vote) | 0.9919 | 0.9930 | 0.9463 | 0.9691 | 0.9962 | 0.0010 |
| Multinomial Naive Bayes | 0.9821 | 0.9272 | 0.9396 | 0.9333 | 0.9911 | 0.0114 |
| Random Forest | 0.9883 | 0.9789 | 0.9329 | 0.9553 | 0.9921 | 0.0031 |

Final model on test set: accuracy 0.9919, precision 0.993, recall 0.9463, F1 0.9691, ROC-AUC 0.9974, confusion matrix {'tn': 965, 'fp': 1, 'fn': 8, 'tp': 141}

## BENCHMARK  -  UCI Phishing Websites (#327)

Rows: 11,055 | Balancing: imblearn.SMOTE | Selected (best validation F1): **Ensemble (RF + HGB, soft vote)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Ensemble (RF + HGB, soft vote) | 0.9756 | 0.9763 | 0.9684 | 0.9723 | 0.9966 | 0.0187 |
| HistGradientBoosting (XGBoost not installed) | 0.9769 | 0.9774 | 0.9704 | 0.9739 | 0.9961 | 0.0179 |
| Random Forest | 0.9729 | 0.9762 | 0.9622 | 0.9692 | 0.9958 | 0.0187 |
| Logistic Regression | 0.9317 | 0.9313 | 0.9133 | 0.9222 | 0.9784 | 0.0536 |

Final model on test set: accuracy 0.9765, precision 0.9803, recall 0.9663, F1 0.9733, ROC-AUC 0.9977, confusion matrix {'tn': 1212, 'fp': 19, 'fn': 33, 'tp': 947}
