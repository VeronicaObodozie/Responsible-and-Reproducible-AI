# Heart Failure Prediction
This is a project for BMEN619.

Heart Failure is the current leading cause of death.

This project looks to develop a model for predicting if an obersvation has heart disease or not in an effort to support early detection and prediction.
Note that although 6 models were tested, only 4 will be used for the project

## Data Exploration
Data was gotten from Kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Understanding the general overview and characteristics of the model
Finding correlations


## Evaluation Criteria

### Data Input and Preprocessing
This includes:
* Removing outlier leaving 917 observations
* Numerical data scaling
* Categorical data encoding
* Spliting data

Maybe changing age data to ranged categories.

### Experimental design

### Metrics for Evaluation
Both performance and fairness metrics were eveluated in this.

Might be broken down into 4 subgroups accross sensitive demographic features of young/old and male/female

#### Performance
* Precision
* Recall/sensitivity
* F1-score

#### Fairness
Based on age and sex sebsitive features:
* Equal odds differennce and ratio.
* Demographic parity difference and ratio.

#### Explainability
* Lime for speed
* shap is good for understanding the global importane (https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html)
https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models


Using python's SHAP package to test the understand the model a bit more. Makes it more explainable using Shapley values

## Model Project
