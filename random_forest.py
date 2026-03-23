# MSE 446
# Random Forest - Tree Ensemble Method

# What householf factors are most likely to affect energy usage and financial struggle?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

#Step 1 - Load cleaned data
X_train = pd.read_csv("clean_x_train.csv")
X_test  = pd.read_csv("clean_x_test.csv")
y_train = pd.read_csv("clean_y_train.csv")
y_test  = pd.read_csv("clean_y_test.csv")

#Step 2 - Split y into regression and classification targets
REGRESSION_TARGETS  = ["DOLLAREL", "DOLLARNG", "DOLLARFO", "DOLLARLP"]
CATEGORICAL_TARGETS = ["SCALEB", "SCALEG", "SCALEE", "PAYHELP", "ENERGYASST", "COLDMA", "HOTMA"]

y_reg_train = y_train[REGRESSION_TARGETS]
y_reg_test  = y_test[REGRESSION_TARGETS]
y_cat_train = y_train[CATEGORICAL_TARGETS]
y_cat_test  = y_test[CATEGORICAL_TARGETS]