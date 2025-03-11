""" 
This code was created by Veronica Obodozie
Date: 5th March 2025

It is part of the BMEN 619: Reproducible and Responsible AI project.

Goal: To choose a model for binary classification prediction model trained through supervised learning.
Input: Heart failure Detection Dataset from Kaggle
Output:Prediction of heart disese or not

Fairness metrics is being used to evaluate
"""

#---------------------------------- IMPORTANT PACKAGES --------------------------------------------#
print('-------------Importing Useful packages------------')
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Data pre-processing
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

# Metrics
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equalized_odds_ratio, demographic_parity_ratio
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score, precision_recall_curve, RocCurveDisplay
#--------------------------------------------------------------------------------------------------------#

#----------------------------------- DATA LOADING -------------------------------------------#
print('-----------------Reading CSV file into dataframe-------------------')
data = pd.read_csv('../heart.csv')
#--------------------------------------------------------------------------------------------------------#

#----------------------------------- DATA PREPROCESSING -------------------------------------------#
print('-------------------Preprocessing-------------------------')
# Scaling
robust_scale = RobustScaler()
standard_scale = StandardScaler()
minmax_scaler = MinMaxScaler()
# Encoding
le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
data['RestingECG'] = le.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le.fit_transform(data['ST_Slope'])

# Scaling Numerical Data
data['Oldpeak'] = robust_scale.fit_transform(data[['Oldpeak']])
data['Age'] = robust_scale.fit_transform(data[['Age']])
data['RestingBP'] = robust_scale.fit_transform(data[['RestingBP']])
data['Cholesterol'] = robust_scale.fit_transform(data[['Cholesterol']])
data['MaxHR'] = robust_scale.fit_transform(data[['MaxHR']])
#--------------------------------------------------------------------------------------------------------#

#----------------------------------- SPLIT DATA -------------------------------------------#
print('------------Spliting Data------------')
target= data['HeartDisease']
features= data.drop('HeartDisease', axis=1)
x_dev, x_test, y_dev, y_test = train_test_split(features, target, test_size = 0.20, random_state = 0)
#--------------------------------------------------------------------------------------------------------#

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
#--------------------------------------------------------------------------------------------------------#

#-------------------------------------- PERFORMANCE METRICS ----------------------------------------#
def metrics(y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot = True)# labels,fmt ='')
    
    # Precision, Recall, F1 score
    print(classification_report(y_test,y_pred))
    print(roc_auc_score(y_test,y_pred))

    # ROC
    RocCurveDisplay.from_predictions(y_test,y_pred)
    plt.title('ROC_AUC_Plot')
    plt.show()
#--------------------------------------------------------------------------------------------------------#

#-------------------------------------- Model ----------------------------------------#
def model(classifier, x_dev, x_test, y_dev, y_test ):
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

    y_pred = classifier.predict(x_test)
    metrics(y_test, y_pred, classifier)
    fairness(x_test, y_test, y_pred)
    
#--------------------------------------------------------------------------------------------------------#
print('--------------------------Support Vector Matrix----------------------------------')
model(SVC(random_state=0, gamma = 10, C=10), x_dev, x_test, y_dev, y_test )

#------------------------------------  K-NEAREST NEIGHBBOR ------------------------------------------#
print('-----------------------------K Nearest Neighbor-----------------------------')
model(KNeighborsClassifier(n_neighbors=5), x_dev, x_test, y_dev, y_test)

#-------------------------------------- RANDOM FOREST CLASSIFIER ----------------------------------------#
print('---------------------------Random Forest Vector--------------------------------')
model(RandomForestClassifier(random_state=0, n_estimators=100, min_samples_split=5, max_depth=10), x_dev, x_test, y_dev, y_test )

#-------------------------------------- ADABOOST ----------------------------------------#
# hyperparameter tuning
print('---------------------------------------Ada Boost-----------------------')
model(AdaBoostClassifier(random_state=0, n_estimators=100, learning_rate=0.001), x_dev, x_test, y_dev, y_test )

#use.get_dummies()