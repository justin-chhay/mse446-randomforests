# MSE 446
# Random Forest - Tree Ensemble Method

# What household factors are most likely to affect energy usage and financial struggle?

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

# Step 1 - Load cleaned data
X_train = pd.read_csv("clean_x_train.csv")
X_test  = pd.read_csv("clean_x_test.csv")
y_train = pd.read_csv("clean_y_train.csv")
y_test  = pd.read_csv("clean_y_test.csv")

# Step 2 - Split y into regression and classification targets
REGRESSION_TARGETS  = ["DOLLAREL", "DOLLARNG", "DOLLARFO", "DOLLARLP"]
CATEGORICAL_TARGETS = ["SCALEB", "SCALEG", "SCALEE", "PAYHELP", "ENERGYASST", "COLDMA", "HOTMA"]

y_reg_train = y_train[REGRESSION_TARGETS]
y_reg_test  = y_test[REGRESSION_TARGETS]
y_cat_train = y_train[CATEGORICAL_TARGETS]
y_cat_test  = y_test[CATEGORICAL_TARGETS]


# Step 3 - Train Random Forest Regressors
reg_models = {}
reg_scores = {}

for col in REGRESSION_TARGETS:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_reg_train[col])
    preds = model.predict(X_test)

    reg_models[col] = model
    reg_scores[col] = {
        "MAE":  mean_absolute_error(y_reg_test[col], preds),
        "RMSE": np.sqrt(mean_squared_error(y_reg_test[col], preds)),
        "R2":   r2_score(y_reg_test[col], preds)
    }

print("=== Regression (Random Forest) ===")
print(pd.DataFrame(reg_scores).T.round(3).to_string())
print('\n')


# Step 4 - Train Random Forest Classifiers
cat_models = {}
cat_scores = {}

for col in CATEGORICAL_TARGETS:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_cat_train[col])
    preds = model.predict(X_test)

    cat_models[col] = model
    cat_scores[col] = {
        "Accuracy":  accuracy_score(y_cat_test[col], preds),
        "F1":        f1_score(y_cat_test[col], preds, average='weighted', zero_division=0),
        "Precision": precision_score(y_cat_test[col], preds, average='weighted', zero_division=0),
        "Recall":    recall_score(y_cat_test[col], preds, average='weighted', zero_division=0)
    }

print("=== Classification (Random Forest) ===")
print(pd.DataFrame(cat_scores).T.round(3).to_string())
print('\n')

# SCALEB - Frequency of reducing or forgoing basic necessities due to home energy bill
# Step 5 - GridSearchCV on SCALEB (primary target of interest)
clf = RandomForestClassifier(class_weight='balanced', random_state=156)
params = {
    'max_depth':    np.arange(5, 30, 5),
    'n_estimators': np.arange(50, 210, 50)
}
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='f1_weighted', # good for imbalanced classes (if many houses are not struggling, accuracy can be misleading)
    #accuracy is misleading for imbalanced classes — f1 balances precision and recall across all classes
    return_train_score=True
)
grid_search.fit(X_train, y_cat_train["SCALEB"])
print("Best params for SCALEB:", grid_search.best_params_)

cat_models["SCALEB"] = grid_search.best_estimator_
tuned_preds = grid_search.best_estimator_.predict(X_test)

#   update cat_scores for SCALEB with tuned model
cat_scores["SCALEB"] = {
    "Accuracy":  accuracy_score(y_cat_test["SCALEB"], tuned_preds),
    "F1":        f1_score(y_cat_test["SCALEB"], tuned_preds, average='weighted', zero_division=0),
    "Precision": precision_score(y_cat_test["SCALEB"], tuned_preds, average='weighted', zero_division=0),
    "Recall":    recall_score(y_cat_test["SCALEB"], tuned_preds, average='weighted', zero_division=0)
}

print("\n=== Tuned SCALEB Metrics ===")
print(f"  Accuracy:  {accuracy_score(y_cat_test['SCALEB'], tuned_preds):.3f}")
print(f"  F1:        {f1_score(y_cat_test['SCALEB'], tuned_preds, average='weighted', zero_division=0):.3f}")
print(f"  Precision: {precision_score(y_cat_test['SCALEB'], tuned_preds, average='weighted', zero_division=0):.3f}")
print(f"  Recall:    {recall_score(y_cat_test['SCALEB'], tuned_preds, average='weighted', zero_division=0):.3f}")
print("\nDetailed Classification Report:")
print(classification_report(y_cat_test["SCALEB"], tuned_preds, zero_division=0))


# Step 6 - Feature Importance — top 10 for DOLLAREL and SCALEB, one combined plot
# extract gini importance from the models and sort
reg_importances = pd.Series(
    reg_models["DOLLAREL"].feature_importances_, index=X_train.columns
).nlargest(10)

cat_importances = pd.Series(
    cat_models["SCALEB"].feature_importances_, index=X_train.columns
).nlargest(10)

print("Top 10 Features for DOLLAREL (Regression):")
print(reg_importances.round(4).to_string())
print("\nTop 10 Features for SCALEB (Classification):")
print(cat_importances.round(4).to_string())

#Visualize the features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

reg_importances.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title("Top 10 Features - DOLLAREL (Regression)")
axes[0].set_xlabel("Mean Decrease in Impurity (Gini importance)")

cat_importances.sort_values().plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title("Top 10 Features - SCALEB (Classification)")
axes[1].set_xlabel("Mean Decrease in Impurity (Gini importance)")

plt.tight_layout()
plt.show()


# Findings
'''
DOLLAREL (Regression) — predicts electricity cost in dollars:
- Square footage (SQFTEST, TOTCSQFT) is the dominant driver
- Household size (NHSLDMEM), dryer use (DRYRUSE), and pool (MONPOOL, SWIMPOOL) also matter

SCALEB (Classification) — predicts difficulty paying energy bills:
- Income (MONEYPY) is by far the strongest predictor
- Age of householder (HHAGE), home quality (DRAFTY), climate (HDD65, CDD65), and education level all contribute

Key observation: Struggle to pay bills is driven primarily by socioeconomic factors (income, age, education),
while actual energy consumption is driven primarily by physical home characteristics (size, appliances, climate).
'''