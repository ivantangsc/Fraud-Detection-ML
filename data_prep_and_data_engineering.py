# %% DATA PREP AND PREPROCESSING 


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


# %%

pd.set_option('display.max_columns', None)
df_bets = pd.read_parquet("df_bets_3years", engine = "pyarrow")
new_bets = pd.read_parquet("df_bets_newday", engine = "pyarrow")
new_bet_ids = new_bets['id'].copy()
df_bets = pd.concat([df_bets, new_bets], ignore_index=True)

################################################################    Data cleaning
# Dropping some columns we don't need, PT values are too updated

df_bets = df_bets.drop('Client ID', axis = 1)
df_bets = df_bets.drop('PT from ME', axis = 1)
df_bets = df_bets.drop('PT', axis = 1)


df_bets = df_bets[df_bets['state'].isin([4, 5])] 

df_bets = df_bets.dropna(subset=['virtual_size']).reset_index(drop =True )

df_bets['virtual_size'].isna().sum()

df_bets["last_update_time"] = pd.to_datetime(df_bets["last_update_time"])
df_bets["last_update_time"] = df_bets["last_update_time"].view('int64') // 10**6
df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"])
df_bets["settled_date"] = pd.to_datetime(df_bets["settled_date"])
df_bets["settled_date"] = df_bets["settled_date"].view('int64') // 10**6



df_bets["price_ratio"] = df_bets["price"] / df_bets["avg_price_matched"]


#   Frequency encoding for competition_name           NOT USED FOR NOW
competition_counts = df_bets['competition_name'].value_counts()
df_bets['competition_counts'] = df_bets['competition_name'].map(competition_counts)

#   Engineered some important features for client accounts, lifetime number bets, average pnl
df_pnlyear = df_bets.groupby("customer_id").agg(      
    total_pnl = ("virtual_profit", "mean"),
    lifetime_bets = ("id", "count")
)
df_bets = df_bets.merge(
    df_pnlyear[['total_pnl', 'lifetime_bets']],
    left_on= 'customer_id',
    right_on =  'customer_id',
    how = 'left'
)

# Creating customer's pnl past 7 days when he placed the bet
df_bets.sort_values(["customer_id", "creation_date"], inplace = True)
rolling_sum = df_bets.groupby("customer_id").rolling("7d", on = "creation_date", closed = "left")["virtual_profit"].sum()
df_bets["pnl_7days"] = rolling_sum.reset_index(drop = True)

df_bets["creation_date"] = df_bets["creation_date"].view('int64') // 10**6 # Back to ms 


# bets and pnl before bet was placed
df_bets = df_bets.sort_values(["customer_id", "creation_date"])
df_bets["num_of_client_bets_before_bet"] = df_bets.groupby("customer_id").cumcount()
df_bets["total_client_profit_before_bet"] = df_bets.groupby("customer_id")["virtual_profit"].cumsum()
df_bets["total_client_stakes"] = df_bets.groupby("customer_id")["virtual_size_matched"].cumsum()

df_bets["total_client_roi"] = df_bets["total_client_profit_before_bet"] / df_bets["total_client_stakes"]

# Time since registration at the time of bet
df_bets["registration_time"] = df_bets.groupby("customer_id")["creation_date"].transform("min")
df_bets["time_since_first_bet"] = df_bets["creation_date"] - df_bets["registration_time"]

# Encoding selection name from String to Float         
df_bets['selection_name'] = df_bets['selection_name'].astype('category')
df_bets['selection_encoded'] = df_bets['selection_name'].cat.codes.astype(float)

# Also event name...            
df_bets['event_name'] = df_bets['event_name'].astype('category')
df_bets['event_name_encoded'] = df_bets['event_name'].cat.codes.astype(float)

# Decided to add customer id as a feature as well. I think that they like using the same account to 
# make multiple fraud so i think it will be helpful

df_bets['customer_id'] = df_bets['customer_id'].astype('category')          
df_bets['customerid_encoded'] = df_bets['customer_id'].cat.codes.astype(float)

df_bets['betplace_matchstart_timedifference'] = df_bets["market_start_time"] - df_bets["creation_date"]

# Boolean conversion for 'side' column values True = 1, False = 0
df_bets['side'] = df_bets['side'].astype(int)
df_bets

#  More Data Engineering

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
df_bets = df_bets.sort_values(["event_name_encoded", "market_id", "selection_encoded", "market_start_time", "creation_date"])
df_bets["num_of_selection_bets_before"] = df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"]).cumcount()



df_bets = df_bets.reset_index(drop=True)
df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"], unit = 'ms')
df_bets["market_start_time"] = pd.to_datetime(df_bets['market_start_time'], unit = 'ms')


df_bets = df_bets.sort_values(["event_name_encoded", "market_id", "selection_encoded", "market_start_time", "creation_date"])
rolling_avg = df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"]).rolling("20min", on = "creation_date", center = True)["price"].mean()

# 10 Mins average price if its after match starts
df_bets["10mins_avg_price"] = rolling_avg.values

# Price range for the selection

df_bets["price_range"] =  (
    df_bets.groupby(["event_name_encoded", "market_id", "selection_encoded", "market_start_time"])["price"].transform(lambda x: x.max() - x.min())
)

df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"])
df_bets["creation_date"] = df_bets["creation_date"].view('int64') // 10**6
df_bets["market_start_time"] = pd.to_datetime(df_bets["market_start_time"])
df_bets["market_start_time"] = df_bets["market_start_time"].view('int64') // 10**6


new_bets = df_bets[df_bets['id'].isin(new_bet_ids)]
df_bets = df_bets[~df_bets['id'].isin(new_bet_ids)] 

df_bets



# 

new_bets.to_parquet('newdaybets', engine = 'pyarrow')
new_bets

#
# %%

df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"], unit = "ms")

split_date = pd.Timestamp('2024-01-31')


df_older = df_bets[df_bets["creation_date"] < split_date]
df_newer = df_bets[df_bets["creation_date"] >= split_date]


df_newer
# %% Fraud ids we have confirmed in the past. Will keep updating.

with open("fraud_betids.txt", "r") as file:
    fraud_betids = [int(line.strip()) for line in file]

fraud_betids
# %%


fraud_bets = df_newer[df_newer["id"].isin(fraud_betids)].reset_index(drop = True)

fraud_bets['fraud'] = 1
fraud_bets
# %%
# CREATING LEGIT BET SAMPLE

non_fraud_bets = df_newer[~df_newer['id'].isin(fraud_bets['id'])]
non_fraud_sample = non_fraud_bets.sample(n = 200000, random_state = 42)

non_fraud_sample['fraud'] = 0


test_sample_new = pd.concat([fraud_bets, non_fraud_sample], ignore_index= True)

test_sample_new["creation_date"] = pd.to_datetime(test_sample_new["creation_date"])
test_sample_new["creation_date"] = test_sample_new["creation_date"].view('int64') // 10**6

test_sample_new.to_parquet('test_sample_new', engine = 'pyarrow')
test_sample_new

# %%

fraud_bets = df_older[df_older["id"].isin(fraud_betids)].reset_index(drop = True)

fraud_bets['fraud'] = 1

# CREATING LEGIT BET SAMPLE

non_fraud_bets = df_older[~df_older['id'].isin(fraud_bets['id'])]
non_fraud_sample = non_fraud_bets.sample(n = 200000, random_state = 42)

non_fraud_sample['fraud'] = 0


test_sample_old = pd.concat([fraud_bets, non_fraud_sample], ignore_index= True)

test_sample_old["creation_date"] = pd.to_datetime(test_sample_old["creation_date"])
test_sample_old["creation_date"] = test_sample_old["creation_date"].view('int64') // 10**6

test_sample_old.to_parquet('test_sample_old', engine = 'pyarrow')
test_sample_old











