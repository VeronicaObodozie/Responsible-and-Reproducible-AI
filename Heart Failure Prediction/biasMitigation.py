#---------------------------------- IMPORTANT PACKAGES --------------------------------------------#
from metrics import *
from model import *
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



#----------------------------------- DATA LOADING -------------------------------------------#
print('-----------------Reading CSV file into dataframe-------------------')
data = pd.read_csv('./heart.csv')

#----------------------------------- DATA PREPROCESSING -------------------------------------------#
print('-------------------Preprocessing-------------------------')
# Outliers
data= data[data['RestingBP'] > 0]

# Scaling
robust_scale = RobustScaler()
standard_scale = StandardScaler()
minmax_scaler = MinMaxScaler()
# Encoding
ohe= OneHotEncoder()
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


#----------------------------------- SPLIT DATA -------------------------------------------#
print('------------Spliting Data------------')
target= data['HeartDisease']
features= data.drop('HeartDisease', axis=1)
x_dev, x_test, y_dev, y_test = train_test_split(features, target, test_size = 0.20, random_state = 0)


print('---------------------------------Applying Preprocessing for fairness: Correlation removal---------------------------')
sensitive_features = ['Age', 'Sex']
# Remove correlation between sensitive features and other features
cr = CorrelationRemover(sensitive_feature_ids=sensitive_features)
x_cr = cr.fit_transform(features)
x_cr_fair=pd.DataFrame(x_cr, columns=['ChestPainType',  'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])
# Create new training data with reduced bias
x_dev_fair, x_test_fair, y_dev_fair, y_test_fair = train_test_split(x_cr_fair, target, test_size = 0.20, random_state = 0)

#
def modelEstimator(classifier, x_train, y_train, x_test):
    sensitive_features = ['Age', 'Sex']
    # Define a new ML estimator that optimizes for fairness
    fair_estimator = ExponentiatedGradient(
        estimator=classifier,  # Your original ML model
        constraints=DemographicParity(),
        eps=0.1  # Fairness constraint violation tolerance
    )

    # Train the fair model
    fair_estimator.fit(
        x_train,
        y_train,
        sensitive_features= np.array(pd.DataFrame(x_train, columns=sensitive_features))
    )

    # POST PROCESSING

    # Create a threshold optimizer
    threshold_optimizer = ThresholdOptimizer(
        estimator=fair_estimator,
        constraints="demographic_parity",
        prefit=True
    )

    # Fit the optimizer
    threshold_optimizer.fit(
        x_train, y_train,
        sensitive_features= np.array(pd.DataFrame(x_train, columns=sensitive_features))
    )

    # Get fair predictions
    fair_predictions = threshold_optimizer.predict(
        x_test,
        sensitive_features= np.array(pd.DataFrame(x_test, columns=sensitive_features))
    )
    return threshold_optimizer, fair_predictions


#----------------------------------- LOGISTIC REGRESSION MODEL----------------------------#
print('--------------------------Logistic Regression---------------------------------')
print('--------------------------With CORRELATION----------------------------------')
classifier_lr, y_pred, fair_pred, fair_model= modelBiasMitigation(LogisticRegression(random_state = 0,C=10,penalty= 'l2'), x_dev, x_test, y_dev, y_test)
print('--------------------------Without CORRELATION----------------------------------')
classifier_lr_fair, y_pred= modelpre(LogisticRegression(random_state = 0,C=10,penalty= 'l2'), x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)

#----------------------------------- SUPPORT VECTOR MATRIX MODEL -------------------------------------------#
print('--------------------------Support Vector Matrix----------------------------------')
print('--------------------------With CORRELATION----------------------------------')
classifier_svc, y_pred, fair_pred, fair_model= modelBiasMitigation(SVC(kernel="linear", random_state=0, gamma = 10, C=10, probability= True), x_dev, x_test, y_dev, y_test)

print('--------------------------Without CORRELATION----------------------------------')
classifier_svc_sex, y_pred= modelpre(SVC(kernel="linear", random_state=0, gamma = 10, C=10, probability= True), x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)


#-------------------------------------- RANDOM FOREST CLASSIFIER ----------------------------------------#
print('---------------------------Random Forest Vector--------------------------------')
classifier_rf, y_pred, fair_pred, fair_model= modelBiasMitigation(RandomForestClassifier(random_state=0, n_estimators=100, min_samples_split=5, max_depth=10), x_dev, x_test, y_dev, y_test)

print('--------------------------Without CORRELATION----------------------------------')
classifier_rf_fair, y_pred= modelpre(RandomForestClassifier(random_state=0, n_estimators=100, min_samples_split=5, max_depth=10),x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)


#-------------------------------------- ADABOOST ----------------------------------------#
# hyperparameter tuning
print('---------------------------------------Ada Boost-----------------------')
classifier_adab, y_pred, fair_pred, fair_model= modelBiasMitigation(AdaBoostClassifier(random_state=0, n_estimators=100, learning_rate=0.001), x_dev, x_test, y_dev, y_test )

print('--------------------------Without CORRELATION----------------------------------')
classifier_adab_fair, y_pred= modelpre(AdaBoostClassifier(random_state=0, n_estimators=100, learning_rate=0.001),x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)


#-------------------------------------- GRADIENT BOOSTING CLASSIFIER ----------------------------------------#
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
print('---------------------------------------GradientBoostingClassifier-----------------------')
classifier_gbc, y_pred, fair_pred, fair_model= modelBiasMitigation(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), x_dev, x_test, y_dev, y_test )

print('--------------------------Without CORRELATION----------------------------------')
classifier_gbc_fair, y_pred= modelpre(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)

# With the base fairness estimation, these gave errors because of sample sizing. So, bias mitigation in and post processing not applied

#-------------------------------------- MULTI-lAYER PERCEPRTION ----------------------------------------#
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised
print('---------------------------------------MLP Classifier----------------------')
# classifier_mlp, y_pred, fair_pred, fair_model= model(MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=9, hidden_layer_sizes=(5, 2), random_state=1), x_dev, x_test, y_dev, y_test )
# #explain(shap.Explainer(classifier_mlp), x_test)

print('--------------------------Without CORRELATION----------------------------------')
classifier_mlp_fair, y_pred= modelpre(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)


#------------------------------------  K-NEAREST NEIGHBBOR ------------------------------------------#
print('-----------------------------K Nearest Neighbor-----------------------------')
# classifier_knn, y_pred, fair_pred, fair_model= model(KNeighborsClassifier(n_neighbors=5), x_dev, x_test, y_dev, y_test)
# #explain(shap.Explainer(classifier_knn), x_test)

print('--------------------------Without CORRELATION----------------------------------')
classifier_knn_fair, y_pred= modelpre(KNeighborsClassifier(n_neighbors=5), x_dev_fair, x_test_fair, y_dev_fair, y_test_fair, x_test)

# Initialize SHAP
shap.initjs()

