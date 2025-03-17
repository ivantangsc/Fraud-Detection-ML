# DATA PREP AND PREPROCESSING 

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
logging.getLogger().setLevel(logging.INFO)
pd.set_option('display.max_columns', None)


def feature_engineering():
    df_bets, new_bet_ids = data_loading()
    timenow= time.time()
    logging.info("Engineering 1/8 Data loaded")
    df_bets = datacleaning(df_bets)
    timenow= time.time()
    logging.info("Engineering 2/8 Datacleaning")
    df_bets = side_related_price_and_profit(df_bets)
    timenow= time.time()
    logging.info("Engineering 3/8 Side related Features")
    df_bets = client_roi(df_bets)
    timenow= time.time()
    logging.info("Engineering 4/8 Client ROI")
    df_bets = match_and_bet_time_difference(df_bets)
    timenow= time.time()
    logging.info("Engineering 5/8 Match time bet time difference")
    df_bets = category_encoding(df_bets)
    timenow= time.time()
    logging.info("Engineering 6/8 Category names encoded")
    df_bets = bet_selection_features(df_bets)
    timenow= time.time()
    logging.info("Engineering 7/8 Bet Selection Features")
    df_bets, new_bets = onehour_avg_price(df_bets, new_bet_ids)
    timenow= time.time()
    logging.info("Engineering 8/8 finished - 1 hour avg price")
    return df_bets, new_bets


#    ################################################################    Data Engineering Functions


def data_loading():
    starttime = time.time()
    df_bets = pd.read_parquet("data/df_bets_1year", engine = "pyarrow")
    new_bets = pd.read_parquet("data/df_bets_newday", engine = "pyarrow")
    new_bet_ids = new_bets['id'].copy()
    df_bets = pd.concat([df_bets, new_bets], ignore_index=True)
    return df_bets, new_bet_ids


def datacleaning (df_bets):
    df_bets = df_bets[df_bets['state'].isin([4, 5])] 
    df_bets = df_bets.dropna(subset=['virtual_size']).reset_index(drop =True )
    df_bets['virtual_size'].isna().sum()
    df_bets["last_update_time"] = pd.to_datetime(df_bets["last_update_time"])
    df_bets["last_update_time"] = df_bets["last_update_time"].astype('int64') // 10**6
    df_bets["matched_date"] = pd.to_datetime(df_bets["matched_date"], unit = "ms")
    df_bets["settled_date"] = pd.to_datetime(df_bets["settled_date"])
    df_bets["settled_date"] = df_bets["settled_date"].astype('int64') // 10**6
    return df_bets


def side_related_price_and_profit (df_bets):
    df_bets['side'] = df_bets['side'].astype(int)
    df_bets['in_play'] = df_bets['in_play'].astype(int)
    df_bets.loc[df_bets["side"] == 0, "virtual_size_matched"] = (
        df_bets.loc[df_bets["side"] == 0, "virtual_size_matched"] *
        (df_bets.loc[df_bets["side"] == 0, "price"] - 1)
    )
    df_bets["profit_ratio"] = df_bets["virtual_profit"] / df_bets["virtual_size_matched"]
    df_bets["profit_ratio"] = df_bets["profit_ratio"].replace([np.inf, -np.inf], 0)
    df_bets.sort_values(["customer_id", "matched_date"], inplace = True)
    df_bets["daily_profit"] = (df_bets.groupby("customer_id")
                              .rolling("1d", on="matched_date", center=True)["virtual_profit"]
                              .sum()
                              .values)
    df_bets["daily_stakes"] = (df_bets.groupby("customer_id")
                              .rolling("1d", on="matched_date", center=True)["virtual_size_matched"]
                              .sum()
                              .values)
    df_bets["daily_roi"] = df_bets["daily_profit"] / df_bets["daily_stakes"]
    df_bets["daily_roi"] = df_bets["daily_roi"].replace([np.inf, -np.inf], 0)
    df_bets["matched_date"] = df_bets["matched_date"].astype('int64') // 10**6 # Back to ms 
    df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"])
    df_bets["creation_date"] = df_bets["creation_date"].astype('int64') // 10**6 
    df_bets.loc[df_bets["side"] == 0, "price"] = 1 + 1 / (
        df_bets.loc[df_bets["side"] == 0, "price"] - 1)
    df_bets["total_matched_ratio"] = df_bets["virtual_size_matched"] / df_bets["total_matched"]
    df_bets["total_matched_ratio"] = df_bets["total_matched_ratio"].replace([np.inf, -np.inf], 0)
    return df_bets

def client_roi (df_bets):
    df_bets = df_bets.sort_values(["customer_id", "matched_date"])
    df_bets["num_of_client_bets_before_bet"] = df_bets.groupby("customer_id").cumcount()
    df_bets["total_client_profit_before_bet"] = df_bets.groupby("customer_id")["virtual_profit"].cumsum()
    df_bets["total_client_stakes"] = df_bets.groupby("customer_id")["virtual_size_matched"].cumsum()
    df_bets["total_client_roi_before_bet"] = df_bets["total_client_profit_before_bet"] / df_bets["total_client_stakes"]
    return df_bets


def match_and_bet_time_difference (df_bets):
    df_bets["registration_time"] = df_bets.groupby("customer_id")["matched_date"].transform("min") # Time since registration at the time of bet
    df_bets["time_since_first_bet"] = df_bets["matched_date"] - df_bets["registration_time"]
    df_bets['betplace_matchstart_timedifference'] = df_bets["market_start_time"] - df_bets["matched_date"]
    df_bets["bet_create_matched_timedifference"] = df_bets["matched_date"] - df_bets["creation_date"]
    return df_bets


def category_encoding (df_bets):
    df_bets['selection_name'] = df_bets['selection_name'].astype('category') # Encoding selection name from String to Float  
    df_bets['selection_encoded'] = df_bets['selection_name'].cat.codes.astype(float) 
    df_bets['event_name'] = df_bets['event_name'].astype('category') # Also event name... 
    df_bets['event_name_encoded'] = df_bets['event_name'].cat.codes.astype(float)
    df_bets['customer_id'] = df_bets['customer_id'].astype('category')          
    df_bets['customerid_encoded'] = df_bets['customer_id'].cat.codes.astype(float)
    return df_bets


def bet_selection_features (df_bets):
    df_selection = df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"]).agg(
        total_selection_bets_num = ("id", "count"),
        total_selection_stakes = ("virtual_size_matched", "sum")
    )
    df_bets = df_bets.merge(
        df_selection[["total_selection_bets_num", "total_selection_stakes"]],
        left_on = ["event_name_encoded", "market_id", "selection_encoded", "market_start_time"],
        right_on = ["event_name_encoded", "market_id", "selection_encoded", "market_start_time"],
        how = 'left'
    )
    df_bets = df_bets.sort_values(["event_name_encoded", "market_id", "selection_encoded", "market_start_time", "matched_date"])
    df_bets["num_of_selection_bets_before"] = df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"]).cumcount()
    return df_bets


def onehour_avg_price (df_bets, new_bet_ids):
    df_bets = df_bets.reset_index(drop=True)
    df_bets["matched_date"] = pd.to_datetime(df_bets["matched_date"], unit = 'ms')
    df_bets["market_start_time"] = pd.to_datetime(df_bets['market_start_time'], unit = 'ms')
    df_bets = df_bets.sort_values(["event_name_encoded", "market_id", "selection_encoded", "market_start_time", "matched_date"])
    rolling_avg = df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"]).rolling("60min", on = "matched_date", center = True)["price"].mean()
    df_bets["1hr_avg_price"] = rolling_avg.values
    df_bets["price_ratio"] = df_bets["price"] / df_bets["1hr_avg_price"]
    df_bets["price_ratio"] = df_bets["price_ratio"].replace([np.inf, -np.inf], 0)
    df_bets["matched_date"] = pd.to_datetime(df_bets["matched_date"])
    df_bets["matched_date"] = df_bets["matched_date"].astype('int64') // 10**6
    df_bets["market_start_time"] = pd.to_datetime(df_bets["market_start_time"])
    df_bets["market_start_time"] = df_bets["market_start_time"].astype('int64') // 10**6
    new_bets = df_bets[df_bets['id'].isin(new_bet_ids)]
    df_bets = df_bets[~df_bets['id'].isin(new_bet_ids)] 
    df_bets.to_parquet("data/1years_modified_data", engine = "pyarrow")
    new_bets.to_parquet('data/newdaybets', engine = 'pyarrow')
    return df_bets, new_bets



#           DATA SAMPLING

def data_sampling (fraud_betids, remove_periods):
    df_bets = pd.read_parquet("data/1years_modified_data", engine = "pyarrow")
    df_bets = df_bets[df_bets["state"] == 4] # Using only winning bets for model training later, proven to improve accuracy via removing unneeded noise
    split_date = pd.Timestamp('2025-1-28')
    df_bets["matched_date"] = pd.to_datetime(df_bets["matched_date"], unit = "ms")
    df_older = df_bets[df_bets["matched_date"] < split_date]
    df_newer = df_bets[df_bets["matched_date"] >= split_date]
    fraud_bets = df_newer[df_newer["id"].isin(fraud_betids)].reset_index(drop = True) # Validation data
    fraud_bets['fraud'] = 1
    non_fraud_bets = df_newer[~df_newer['id'].isin(fraud_bets['id'])]
    non_fraud_bets = non_fraud_bets.copy()  
    non_fraud_bets['year_month'] = non_fraud_bets['matched_date'].dt.to_period('M')
    non_fraud_bets = non_fraud_bets[~non_fraud_bets['year_month'].isin(remove_periods)]
    non_fraud_sample = non_fraud_bets.sample(n = 200000, random_state = 42)
    non_fraud_sample['fraud'] = 0
    validation_data = pd.concat([fraud_bets, non_fraud_sample], ignore_index= True)
    fraud_bets = df_older[df_older["id"].isin(fraud_betids)].reset_index(drop = True)
    fraud_bets['fraud'] = 1
    non_fraud_bets = df_older[~df_older['id'].isin(fraud_bets['id'])]
    non_fraud_bets = non_fraud_bets.copy()  # Create an explicit copy
    non_fraud_bets['year_month'] = non_fraud_bets['matched_date'].dt.to_period('M')
    non_fraud_bets = non_fraud_bets[~non_fraud_bets['year_month'].isin(remove_periods)]
    non_fraud_sample = non_fraud_bets.sample(n = 200000, random_state = 53)
    non_fraud_sample['fraud'] = 0
    train_test_sample = pd.concat([fraud_bets, non_fraud_sample], ignore_index= True)
    train_test_sample.to_parquet('data/train_test_sample', engine = 'pyarrow')
    validation_data.to_parquet('data/validation_data', engine = 'pyarrow')
    return validation_data, train_test_sample

def get_fraud_betids():                     # Fraud ids we have confirmed in the past. Will keep updating.
    with open("data/fraud_betids.txt", "r") as file: 
        fraud_betids = [int(line.strip()) for line in file]
    remove_periods = [                      # Some months have noticiably less fraud bets, due to mannual inefficiency, so I am removing these months
        pd.Period('2022-04', freq='M'),
        pd.Period('2022-05', freq='M'),
        pd.Period('2022-06', freq='M'),
        pd.Period('2022-07', freq='M'),
        pd.Period('2022-05', freq='M'),
        pd.Period('2023-05', freq='M'),
        pd.Period('2023-08', freq='M'),
        pd.Period('2023-10', freq='M'),
        pd.Period('2023-11', freq='M'),
        pd.Period('2023-12', freq='M'),
        pd.Period('2024-06', freq='M'),
        pd.Period('2024-07', freq='M'),
        pd.Period('2024-10', freq='M'),
        pd.Period('2025-03', freq='M'),
    ]
    return fraud_betids, remove_periods




