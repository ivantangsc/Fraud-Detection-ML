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
import catboost
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import sys
import numpy as np
import shap
from tempfile import mkdtemp
logging.getLogger().setLevel(logging.INFO)
from catboost import CatBoostClassifier

# %%

test_sample = pd.read_parquet("test_sample", engine = "pyarrow")

numerical_sample = test_sample[["fraud",  "customer_id", "price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
                                "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
                                "handicap", "virtual_profit", "virtual_size", "virtual_size_matched", 
                                "side", "state", "betplace_matchstart_timedifference",
                                "num_of_selection_bets_before"]]

numerical_sample.info()



y = numerical_sample["fraud"]

X = numerical_sample.drop("fraud", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



cat_model = CatBoostClassifier(random_state = 42)

cat_model.fit(X_train, y_train, cat_features = ["customer_id"])

y_pred = cat_model.predict(X_test)

print(classification_report(y_test, y_pred))

training_score = cat_model.score(X_train, y_train)
test_score = cat_model.score(X_test, y_test)

print("Training score:", training_score)
print("Test score:", test_score)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("True Negatives:", tn)
print("False Positives", fp)
print("False Negatives", fn)
print("True Positives", tp)

importances = cat_model.feature_importances_

importance_df = pd.DataFrame({"feature": X_train.columns, "importance": importances}).sort_values(by = 'importance', ascending =False,)

importance_df

plt.figure(figsize = (10,6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('CatBoost Feature Importance')
plt.gca().invert_yaxis()
bars = plt.barh(importance_df['feature'], importance_df['importance'])
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', 
             va='center', ha='left', color='black', fontsize=10)
plt.show()

# %%

last_week_bets = pd.read_parquet('newdaybets', engine = 'pyarrow')
customer_ids = last_week_bets[["id", "settled_date"]]
last_week_bets = last_week_bets[["customer_id", "price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
                                "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
                                "handicap", "virtual_profit", "virtual_size", "virtual_size_matched", 
                                "side", "state", "betplace_matchstart_timedifference",
                                "num_of_selection_bets_before"]]
last_week_bets



predicted_labels = cat_model.predict(last_week_bets)

predicted_probs = cat_model.predict_proba(last_week_bets)


last_week_bets["predicted_labels"] = predicted_labels

predicted_probs_df = pd.DataFrame(predicted_probs, 
                                  index=last_week_bets.index,
                                  columns=["prob_0", "prob_1"])


last_week_bets = pd.concat([last_week_bets, predicted_probs_df], axis=1)

result = last_week_bets.merge(customer_ids, left_index=True, right_index=True)


result = result[result["prob_1"] >= 0.5]




result["settled_date"] = pd.to_datetime(result["settled_date"], unit = 'ms')

result = result.sort_values(by = "prob_1", ascending=False)

result.to_csv("fraud_bets.csv", index=False)
result

# %%

from shap import summary_plot

last_week_bets = pd.read_parquet('1weekbets', engine = 'pyarrow')
customer_ids = last_week_bets[['customer_id', "id", "settled_date"]]
last_week_bets = last_week_bets[["customer_id", "price", "avg_price_matched", "10mins_avg_price", "num_of_client_bets_before_bet",
                                "total_selection_bets_num", "time_since_first_bet", "total_client_profit_before_bet",
                                "handicap", "virtual_profit", "virtual_size", "virtual_size_matched", 
                                "side", "state", "betplace_matchstart_timedifference",
                                "num_of_selection_bets_before"]]

explainer = shap.TreeExplainer(cat_model)
feature_columns = last_week_bets.columns
shap_values = explainer.shap_values(last_week_bets)

shap,summary_plot(shap_values, last_week_bets)