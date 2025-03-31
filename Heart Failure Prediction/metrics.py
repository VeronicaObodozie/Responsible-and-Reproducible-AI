print('-------------Importing Useful packages------------')
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
import os
from PIL import Image
import matplotlib.pyplot as plt

# Metrics
# fairness
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equalized_odds_ratio, demographic_parity_ratio, MetricFrame
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_auc_score, precision_recall_curve, RocCurveDisplay



#-------------------------------------- PERFORMANCE METRICS ----------------------------------------#
# Metrics Function
def metrics(y_test, y_pred):
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    
    # Precision, Recall, F1 score
    print(classification_report(y_test,y_pred))
    print(roc_auc_score(y_test,y_pred))

    # ROC
    RocCurveDisplay.from_predictions(y_test,y_pred)
    plt.title('ROC_AUC_Plot')
    plt.show()

#------------------------------------ FAIRNESS EVALUATION ------------------------------------------#
def fairness(x_test, y_test, y_pred):
    age_sensitive= x_test['Age']
    sex_sensitive = x_test['Sex']
    sensitive_features = ['Age', 'Sex']
    # Demographic Parity
    age_dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=age_sensitive)
    sex_dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sex_sensitive)
    age_dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=age_sensitive)
    sex_dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=sex_sensitive)
    print(f'The Age Demographic Parity Difference is: {age_dpd}')
    print(f'The Age Demographic Parity Ratio is: {age_dpr}')
    print(f'The Sex Demographic Parity Difference is: {sex_dpd}')
    print(f'The Sex Demographic Parity Ratio is: {sex_dpr}')

    # equalized_odds
    age_eod = equalized_odds_difference(y_test, y_pred, sensitive_features=age_sensitive)
    sex_eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sex_sensitive)
    age_eor = equalized_odds_ratio(y_test, y_pred, sensitive_features=age_sensitive)
    sex_eor = equalized_odds_ratio(y_test, y_pred, sensitive_features=sex_sensitive)
    print(f'The Age Equalized Odds Difference is: {age_eod}')
    print(f'The Age Equalized Odds Ratio is: {age_eor}')
    print(f'The Sex Equalized Odds Difference is: {sex_eod}')
    print(f'The Sex Equalized Odds Ratio is: {sex_eor}')

def performFair(x_test, y_test, y_pred):
    
    sensitive_features = ['Age', 'Sex']
    metric_frame = MetricFrame(
        metrics={
            "Area Under the curve": roc_auc_score,
            "demographic_parity_difference": demographic_parity_difference,
            "demographic_parity_ratio": demographic_parity_ratio,
            "equalized_odds_difference": equalized_odds_difference,
            "equalized_odds_ratio": equalized_odds_ratio,
            "Classification Report": classification_report,
        },
        sensitive_features=x_test[sensitive_features],
        y_true=y_test,
        y_pred=y_pred,
    )
    print(metric_frame.overall)
    print(metric_frame.by_group)
    metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 1],
        legend=False,
        figsize=[12, 8],
        title="Accuracy and selection rate by group",
    )
    print('---------------------------------Both Sensitive Features Applied--------------------------------')
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=x_test[sensitive_features])
    dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=x_test[sensitive_features])
    print(f'The Demographic Parity Difference is: {dpd}')
    print(f'The Demographic Parity Ratio is: {dpr}')
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=x_test[sensitive_features])
    eor = equalized_odds_ratio(y_test, y_pred, sensitive_features=x_test[sensitive_features])
    print(f'The Equalized Odds Difference is: {eod}')
    print(f'The Equalized Odds Ratio is: {eor}')