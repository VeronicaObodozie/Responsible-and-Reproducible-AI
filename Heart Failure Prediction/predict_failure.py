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
from metrics import *
from model import *
print('-------------Importing Useful packages------------')
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Data pre-processing
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from fairlearn.preprocessing import CorrelationRemover


# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equalized_odds_ratio, demographic_parity_ratio, MetricFrame
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_auc_score, RocCurveDisplay, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix

# Bias Mitigation
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Explainability
import shap
#--------------------------------------------------------------------------------------------------------#

#----------------------------------- DATA LOADING -------------------------------------------#
print('-----------------Reading CSV file into dataframe-------------------')
data = pd.read_csv('./heart.csv')
#--------------------------------------------------------------------------------------------------------#

#----------------------------------- DATA PREPROCESSING -------------------------------------------#
print('-------------------Preprocessing-------------------------')
# Outliers
data= data[data['RestingBP'] > 0]

# Scaling
robust_scale = RobustScaler()

# Encoding
le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
data['RestingECG'] = le.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le.fit_transform(data['ST_Slope'])
data['Age'] = np.where(data['Age'].between(0,40), 0, data['Age']) # Young
data['Age'] = np.where(data['Age'].between(41,60), 1, data['Age']) # Middle Aged
data['Age'] = np.where(data['Age'].between(61,77), 2, data['Age']) # Seniors

# Scaling Numerical Data
data['Oldpeak'] = robust_scale.fit_transform(data[['Oldpeak']])
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

#-------------------------------------EXPLAINABILITY--------------------------------------#
shap.initjs()

#--------------------------------------------------------------------------------------------------------#
#----------------------------------- SUPPORT VECTOR MATRIX MODEL -------------------------------------------#
print('--------------------------Support Vector Matrix----------------------------------')
classifier_svc, y_pred = model(SVC(kernel="linear", random_state=0, gamma = 10, C=10, probability= True), x_dev, x_test, y_dev, y_test)
explain(classifier_svc, x_dev, x_test)

#-------------------------------------- RANDOM FOREST CLASSIFIER ----------------------------------------#
print('---------------------------Random Forest Vector--------------------------------')
classifier_rf, y_pred = model(RandomForestClassifier(random_state=0, n_estimators=100, min_samples_split=5, max_depth=10), x_dev, x_test, y_dev, y_test)
explain(classifier_rf, x_dev, x_test)

#-------------------------------------- ADABOOST ----------------------------------------#
print('---------------------------------------Ada Boost-----------------------')
classifier_adab, y_pred = model(AdaBoostClassifier(random_state=0, n_estimators=100, learning_rate=0.001), x_dev, x_test, y_dev, y_test )
explain(classifier_adab, x_dev, x_test)

#-------------------------------------- MULTI-lAYER PERCEPRTION ----------------------------------------#
print('---------------------------------------MLP Classifier----------------------')
classifier_mlp, y_pred= model(MLPClassifier(solver='lbfgs', max_iter=500, hidden_layer_sizes=(50), random_state=0), x_dev, x_test, y_dev, y_test )
explain(classifier_mlp, x_dev, x_test)

#------------------------------------  K-NEAREST NEIGHBBOR ------------------------------------------#
print('-----------------------------K Nearest Neighbor-----------------------------')
classifier_knn, y_pred= model(KNeighborsClassifier(n_neighbors=5), x_dev, x_test, y_dev, y_test)
explain(classifier_knn, x_dev, x_test)
