# %%

import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
import time
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import concurrent.futures
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
logging.getLogger().setLevel(logging.INFO)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import sys
import numpy as np
import shap

logging.getLogger().setLevel(logging.INFO)

# %%

test_sample = pd.read_parquet('test_sample', engine = 'pyarrow')
test_sample.info()

numerical_sample = test_sample.select_dtypes(exclude = ['object', 'category'])

# %% TEST SPLIT


y = numerical_sample['fraud']
X = numerical_sample.drop('fraud', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %%

xgb_model = XGBClassifier(random_state = 42)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred))

train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
print(f"Training score {train_score}")
print(f"Test score {test_score}")

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv = 5)
print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean()}")
# %%

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# %%

importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values(by= 'importance', ascending = False)

plt.figure(figsize = (10,6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Xgboost Feature Importance')
plt.gca().invert_yaxis()
bars = plt.barh(importance_df['feature'], importance_df['importance'])
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', 
             va='center', ha='left', color='black', fontsize=10)
plt.show()





# %% REAL TEST

last_week_bets = pd.read_parquet('1weekbets', engine = 'pyarrow')
customer_ids = last_week_bets[['customer_id']]
last_week_bets = last_week_bets.select_dtypes(exclude = ['object', 'category'])
last_week_bets

# %%

predicted_labels = xgb_model.predict(last_week_bets)

predicted_probs = xgb_model.predict_proba(last_week_bets)

# %%

last_week_bets["predicted_labels"] = predicted_labels

predicted_probs_df = pd.DataFrame(predicted_probs, 
                                  index=last_week_bets.index,
                                  columns=["prob_0", "prob_1"])


last_week_bets = pd.concat([last_week_bets, predicted_probs_df], axis=1)

result = last_week_bets.merge(customer_ids, left_index=True, right_index=True)


result = result[result["prob_1"] >= 0.5]


# %%

result = result.sort_values(by = "prob_1", ascending=False)

result.to_csv("fraud_bets.csv", index=False)
result

# %% SHAP
from shap import summary_plot

last_week_bets = pd.read_parquet('1weekbets', engine = 'pyarrow')
customer_ids = last_week_bets[['customer_id']]
last_week_bets = last_week_bets.select_dtypes(exclude = ['object', 'category'])

explainer = shap.TreeExplainer(xgb_model)
feature_columns = last_week_bets.columns
shap_values = explainer.shap_values(last_week_bets[feature_columns])

shap,summary_plot(shap_values, last_week_bets[feature_columns])