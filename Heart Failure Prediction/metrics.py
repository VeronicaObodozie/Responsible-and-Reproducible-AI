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
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_auc_score, RocCurveDisplay, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix

import shap

#-------------------------------------- PERFORMANCE and FAIRNESS METRICS ----------------------------------------#
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def performFair(sensitive_feature, y_test, y_pred):

    metric_frame = MetricFrame(
        metrics={
            "Recall or Sensitivity": recall_score,
            "Precision": precision_score,
            "F1 Score": f1_score,
            "Balanced Accuracy": balanced_accuracy_score,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features= sensitive_feature,
    )
    print('OVERALL METRICS')
    print(metric_frame.overall)
    print('METRIC BY GROUP')
    print(metric_frame.by_group)
    # print('CONFIDENCE INTERVALS')
    # print(metric_frame.by_group_ci)
    metric_frame.by_group.plot.bar(
        subplots=True,
        # x= ['Young Men', 'Young Women', 'Middle Aged Men', 'Middle Aged Women', 'Senior Men', 'Senior Women'],
        layout=[4, 1],
        legend=False,
        figsize=[12, 8],
        title="Accuracy and selection rate by group",
    )
    print('---------------------------------Both Sensitive Features Applied--------------------------------')
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
    dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_feature)
    print(f'The Demographic Parity Difference is: {dpd}')
    print(f'The Demographic Parity Ratio is: {dpr}')
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_feature)
    eor = equalized_odds_ratio(y_test, y_pred, sensitive_features=sensitive_feature)
    print(f'The Equalized Odds Difference is: {eod}')
    print(f'The Equalized Odds Ratio is: {eor}')


#-------------------------------------- PERFORMANCE METRICS ----------------------------------------#
# Metrics Function
def metrics(y_test, y_pred):
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    
    # Precision, Recall, F1 score
    print(classification_report(y_test,y_pred))
    print(f'Specificity: ', {specificity(y_test, y_pred)})
    print(roc_auc_score(y_test,y_pred))

    # ROC
    RocCurveDisplay.from_predictions(y_test,y_pred)
    plt.title('ROC_AUC_Plot')
    plt.show()

#------------------------------------ FAIRNESS EVALUATION ------------------------------------------#
def fairness(x_test, y_test, y_pred):
    age_sensitive= x_test['Age']
    sex_sensitive = x_test['Sex']
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


#-------------------------------------EXPLAINABILITY--------------------------------------#
def explain(classifier, x_dev, X_test):
    explainer = shap.Explainer(classifier.predict, x_dev)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names= classifier.classes_)