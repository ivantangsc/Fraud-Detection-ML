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
from tempfile import mkdtemp
logging.getLogger().setLevel(logging.INFO)

# %% FOCUS IS RECALL! HIGH TRUE POSITIVE, LOW FALSE NEGATIVES, FALSE POSITIVE IS OK.

pd.set_option('display.max_columns', None)
test_sample_old = pd.read_parquet('test_sample_old', engine = 'pyarrow')
test_sample_new = pd.read_parquet('test_sample_new', engine = 'pyarrow')
test_sample_new['market_id'] = pd.to_numeric(test_sample_new['market_id'], errors='coerce')

test_sample_new = test_sample_new.dropna(subset=['market_id'])

test_sample_old['market_id'] = pd.to_numeric(test_sample_old['market_id'], errors='coerce')

test_sample_old = test_sample_old.dropna(subset=['market_id'])

test_sample_new["10minprice_ratio"] = test_sample_new["price"] / test_sample_new["10mins_avg_price"]
test_sample_old["10minprice_ratio"] = test_sample_old["price"] / test_sample_old["10mins_avg_price"]

test_sample_new["profit_ratio"] = test_sample_new["virtual_profit"] / test_sample_new["virtual_size_matched"]
test_sample_old["profit_ratio"] = test_sample_old["virtual_profit"] / test_sample_old["virtual_size_matched"]



#%%
# numerical_sample = test_sample[["fraud", "price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
#                                 "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
#                                 "handicap", "virtual_profit", "virtual_size", "virtual_size_matched", 
#                                 "side", "state", "betplace_matchstart_timedifference", 
#                                 "num_of_selection_bets_before"]]


numerical_sample = test_sample_old[["fraud", "num_of_client_bets_before_bet", "profit_ratio", "10minprice_ratio", 
                                    "total_selection_bets_num", "total_client_profit_before_bet", "state", "side", "betplace_matchstart_timedifference",
                                    "handicap", "Currency", "num_of_selection_bets_before"]]

numerical_sample.info()

#  TEST SPLIT


y = numerical_sample['fraud']
X = numerical_sample.drop('fraud', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#                        GRID SEARCH 

# warnings.filterwarnings('ignore')
# cachedir = mkdtemp()

# pipeline = Pipeline([
#     ('classifier', XGBClassifier(random_state = 42, use_label_encoder = False, eval_metric = 'logloss'))
# ])

# param_grid = {
#     'classifier__n_estimators' : [600,700, 800],
#     'classifier__max_depth': [7, 8, 9],
#     'classifier__learning_rate': [0.09, 0.1, 0.2],
#     'classifier__subsample': [0.9,1.0, 1.1]
# }

# grid_search = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv =5,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     verbose = 1
# )


# grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)
# print('Best cross-validation score:', grid_search.best_score_)


# # 

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

xgb_model = XGBClassifier(random_state = 42, enable_categorical = True)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred))

train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
print(f"Training score {train_score}")
print(f"Test score {test_score}")

# cv_scores = cross_val_score(xgb_model, X_train, y_train, cv = 5)
# print(f"Mean CV score: {cv_scores.mean()}")
# 

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")



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

# %% VALIDATION TEST WITH NEW DATA

numerical_sample_new = test_sample_new[["fraud", "num_of_client_bets_before_bet", "profit_ratio", "10minprice_ratio", 
                                        "total_selection_bets_num", "total_client_profit_before_bet", "state", "side", "betplace_matchstart_timedifference",
                                        "handicap", "Currency", "num_of_selection_bets_before"]]


y_recent = numerical_sample_new["fraud"]
X_recent = numerical_sample_new.drop("fraud", axis =1)

y_recent_pred = xgb_model.predict(X_recent)

print("=== Evaluation on Recent (Validation) Data ===")
print(classification_report(y_recent, y_recent_pred))
cm_recent = confusion_matrix(y_recent, y_recent_pred)
tn, fp, fn, tp = cm_recent.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

cv_scores = cross_val_score(xgb_model, X_recent, y_recent, cv = 5)

print(f"Mean CV score: {cv_scores.mean()}")
































# %% REAL TEST
pd.set_option('display.max_columns', None)
last_week_bets = pd.read_parquet('newdaybets', engine = 'pyarrow')
last_week_bets

customer_ids = last_week_bets[['customer_id', "id", "settled_date", "creation_date", "market_start_time","last_modified", "virtual_size", "virtual_size_matched", "virtual_profit"]]
last_week_bets = last_week_bets[["price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
                                "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
                                "handicap",
                                "side", "state", "betplace_matchstart_timedifference",
                                "num_of_selection_bets_before"]]
last_week_bets



predicted_labels = xgb_model.predict(last_week_bets)

predicted_probs = xgb_model.predict_proba(last_week_bets)


last_week_bets["predicted_labels"] = predicted_labels

predicted_probs_df = pd.DataFrame(predicted_probs, 
                                  index=last_week_bets.index,
                                  columns=["prob_0", "prob_1"])


last_week_bets = pd.concat([last_week_bets, predicted_probs_df], axis=1)

result = last_week_bets.merge(customer_ids, left_index=True, right_index=True)

result = result[["id", "customer_id", "prob_1", "creation_date", "market_start_time", "last_modified", "price", 
                 "side", "virtual_size", "virtual_size_matched", "virtual_profit"]]

result = result[result["prob_1"] >= 0.7]

result["creation_date"] = pd.to_datetime(result["creation_date"], unit = "ms")
result["market_start_time"] = pd.to_datetime(result["market_start_time"], unit = "ms")
result["last_modified"] = pd.to_datetime(result["last_modified"], unit = "ms")



# result["market_start_time"] = pd.to_datetime(result["market_start_time"], unit = 'ms')
# result["settled_date"] = pd.to_datetime(result["settled_date"], unit = 'ms')

result = result.sort_values(by = "prob_1", ascending=False)

result.to_csv("fraud_bets.csv", index=False)
result

# %% SHAP
from shap import summary_plot

last_week_bets = pd.read_parquet('newdaybets', engine = 'pyarrow')
customer_ids = last_week_bets[['customer_id', "id", "settled_date"]]
last_week_bets = last_week_bets[["price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
                                "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
                                 "handicap",
                                "side", "state", "betplace_matchstart_timedifference",
                                "num_of_selection_bets_before"]]

explainer = shap.TreeExplainer(xgb_model)
feature_columns = last_week_bets.columns
shap_values = explainer.shap_values(last_week_bets)

shap,summary_plot(shap_values, last_week_bets)