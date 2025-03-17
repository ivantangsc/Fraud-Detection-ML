# %%

from utils.getnewbets import get_secrets, get_new_daily_bets, transform_new_bets
from utils.data_prep_and_data_engineering import feature_engineering, get_fraud_betids, data_sampling
from utils.model_training_testing import predict_new_daily_bets
from utils.model_training_testing import training_and_testing_model
from utils.model_training_testing import grid_search
from utils.model_training_testing import validation_test
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
from dotenv import load_dotenv
logging.getLogger().setLevel(logging.INFO)

todays_date = datetime.datetime(2025, 3, 16, 20, 0)

# %% ETL > Feature Engineering > Machine Learning for new bets Workflow #### Once every Day

# ETL
starttime = time.time()
logging.info("Start ETL process for new daily bets ")
APIKEY_pro, USER_NAME_pro, PASSWORD_pro, referer, url_pro, url_bets = get_secrets() 
get_new_daily_bets(APIKEY_pro, USER_NAME_pro, PASSWORD_pro, referer, url_pro, url_bets, todays_date)
df_merged = transform_new_bets()  

# Feature Engineering
logging.info("Begin Feature engineering ---%.2f seconds ---" % (time.time() - starttime))
df_bets, new_bets = feature_engineering()
logging.info("Feature engineering finished ---%.2f seconds ---" % (time.time() - starttime))

# Applying the model on new daily bets
result = predict_new_daily_bets()
logging.info(f"Number of unique customer accounts flagged today {result['customer_id'].nunique()}")
logging.info("WorkFlow finished ---%.2f seconds ---" % (time.time() - starttime))
result




#%%             Merging new daily bets with all bets each day, BUT after new bets are predicted by the fraud model

df_merged = pd.read_parquet("data/df_bets_newday", engine = "pyarrow")
df_mergedyear = pd.read_parquet("data/df_bets_1year", engine = "pyarrow")
df_mergedyear.sort_values(by = "last_update_time", ascending = False)
# %%
df_mergedyear = pd.concat([df_mergedyear, df_merged], ignore_index = True) 
df_mergedyear.to_parquet("data/df_bets_1year", engine = "pyarrow")
df_mergedyear.sort_values(by = "last_update_time", ascending= False)



# %% Model Training and Testing #### Once every Month

# Data Sampling
fraud_betids, remove_periods = get_fraud_betids()
validation_data, train_test_sample = data_sampling (fraud_betids, remove_periods)

# Train and Test Split
train_test_sample = pd.read_parquet('data/train_test_sample', engine = 'pyarrow')
validation_data = pd.read_parquet('data/validation_data', engine = 'pyarrow')
train_test_sample.info()

xgb_model, X_train, X_test, y_train, y_test = training_and_testing_model(train_test_sample)

# GRID SEARCH
best_model = grid_search(X_train, X_test, y_train, y_test)

# VALIDATION TEST
y_recent_pred = validation_test(best_model, validation_data)

