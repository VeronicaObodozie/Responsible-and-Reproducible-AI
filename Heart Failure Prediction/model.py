#---------------------------------- IMPORTANT PACKAGES --------------------------------------------#
print('-------------Importing Useful packages------------')
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
import os
from PIL import Image
import matplotlib.pyplot as plt

# Data pre-processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from fairlearn.preprocessing import CorrelationRemover

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
# fairness
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equalized_odds_ratio, demographic_parity_ratio, MetricFrame
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_auc_score, RocCurveDisplay, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix

# Bias Mitigation
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Explainability
import shap


#-------------------------------------- Model ----------------------------------------#
# in and post process mitiagation techniques
def model(classifier, x_dev, x_test, y_dev, y_test):
    kf=StratifiedKFold(n_splits=9)
    for fold , (train,validate) in enumerate(kf.split(X=x_dev,y=y_dev)):
        
        X_train=x_dev.iloc[train]
        y_train=y_dev.iloc[train]
        
        X_valid=x_dev.iloc[validate]
        y_valid=y_dev.iloc[validate]
        
        classifier.fit(X_train,y_train)
        
        y_pred=classifier.predict(X_valid)
        print(f"The fold is : {fold} : ")
        print(classification_report(y_valid,y_pred))
        acc=roc_auc_score(y_valid,y_pred)
        print(f"The accuracy for Fold {fold+1} : {acc}")
        fair_model, fair_pred = modelEstimator(classifier, X_train, y_train, X_valid)
        accf=roc_auc_score(y_valid,fair_pred)
        print(f"The FAIR accuracy for Fold {fold+1} : {accf}")
        pass
    print('-----------------------Base Model-------------------------')
    y_pred = classifier.predict(x_test)
    #performFair(x_test, y_test, y_pred)
    y_t = y_test.to_numpy()
    performFair(pd.DataFrame(x_test, columns=["Age", "Sex"]), y_t, y_pred)
    metrics(y_test, y_pred)
    # fairness(x_test, y_test, y_pred)
    print('-----------------------Fair Model-------------------------')
    fair_pred = fair_model.predict(x_test, sensitive_features=x_test[sensitive_features])
    #performFair(x_test, y_test, y_pred)
    performFair(pd.DataFrame(x_test, columns=["Age", "Sex"]), y_t, fair_pred)
    metrics(y_test, fair_pred)
    # fairness(x_test, y_test, fair_pred)
    return classifier, y_pred, fair_pred, fair_model
    
#--------------------------------------------------------------------------------------------------------#

#-------------------------------------- Correlation Model ----------------------------------------#
# pre processing mtigation techniques
def modelpre(classifier, x_dev, x_test, y_dev, y_test, xt):
    kf=StratifiedKFold(n_splits=9)
    for fold , (train,validate) in enumerate(kf.split(X=x_dev,y=y_dev)):
        
        X_train=x_dev.iloc[train]
        y_train=y_dev.iloc[train]
        
        X_valid=x_dev.iloc[validate]
        y_valid=y_dev.iloc[validate]
        
        classifier.fit(X_train,y_train)
        
        y_pred=classifier.predict(X_valid)
        print(f"The fold is : {fold} : ")
        print(classification_report(y_valid,y_pred))
        acc=roc_auc_score(y_valid,y_pred)
        print(f"The accuracy for Fold {fold+1} : {acc}")
        pass
    print('-----------------------Base Model-------------------------')
    y_pred = classifier.predict(x_test)
    #performFair(x_test, y_test, y_pred)
    metrics(y_test, y_pred)
    fairness(xt, y_test, y_pred)
    return classifier, y_pred
    
#--------------------------------------------------------------------------------------------------------#