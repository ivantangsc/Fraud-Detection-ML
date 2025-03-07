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
from dotenv import load_dotenv
logging.getLogger().setLevel(logging.INFO)


# %%

load_dotenv(dotenv_path="data/.env")
APIKEY_pro = os.getenv("APIKEY_pro")
USER_NAME_pro = os.getenv("USER_NAME_pro")
PASSWORD_pro = os.getenv("PASSWORD_pro")
referer = os.getenv("referer")
url_pro = os.getenv("url_pro")
url_bets = os.getenv("url_bets")

current_time = datetime.datetime.now()
start_of_today = datetime.datetime(current_time.year, current_time.month, current_time.day)
start_of_yesterday = start_of_today - datetime.timedelta(days=0)
end_checkpoint = int(time.mktime(start_of_yesterday.timetuple())) * 1000
desired_date = datetime.datetime(2025, 3, 5, 20, 0)
desired_checkpoint = int(time.mktime(desired_date.timetuple())) * 1000
print(end_checkpoint)
print(desired_checkpoint)
day = 6
chunk_index = 0

#               Get Login Token

token = get_token(APIKEY_pro, USER_NAME_pro, PASSWORD_pro, referer, url_pro)

# %%            Get new daily bets

get_new_daily_bets(token, APIKEY_pro, referer, url_bets, desired_checkpoint, end_checkpoint, day, chunk_index)

# %%            Modify all bet currencies to the same (HKD) + New daily Bets ready for Modelling

df_merged = modify_bets_currencies()

# %%            Merging new daily bets with all bets each day, BUT after new bets are used in fraud model

df_mergedyear = pd.read_parquet("data/df_bets_1year", engine = "pyarrow")
df_mergedyear = pd.concat([df_mergedyear, df_merged], ignore_index = True) 

df_mergedyear.to_parquet("data/df_bets_1year", engine = "pyarrow")









#%%                    ###############      FUNCTIONS

def get_token(APIKEY_pro, USER_NAME_pro, PASSWORD_pro, referer, url_pro):
    headers = {
        "X-API_KEY": APIKEY_pro,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-GB,en;q=0.5",
        "Content-type": "Application/JSON",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
        "Referer": referer,
        "Cookie": "date_lang=en"
    }
    data = {
        "username": USER_NAME_pro,
        "password": PASSWORD_pro
    }
    response = requests.post(url_pro, headers=headers, json=data, verify=False)
    response.text
    token = response.json()["token"]
    return token




def get_new_daily_bets(token, APIKEY_pro, referer, url_bets, desired_checkpoint, end_checkpoint, day, chunk_index):
    headers = {
        "Authorization": f"Bearer {token}",
        "X-API_KEY": APIKEY_pro,
        "Content-type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-GB,en;q=0.5",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
        "Referer": referer,
        "Cookie": "date_lang=en"
        }
    all_bets = []
    max_workers = 40
    CHUNK_SIZE = 400
    count = 0
    for chunk_start in range(0, day, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, day)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
            for i in range(chunk_start, chunk_end):
                data= {
                    "last_update_time":desired_checkpoint,
                    "to_last_update_time":end_checkpoint,
                    "client_accountId": None,
                    "state": [4,5]
                }
                desired_checkpoint  -= 14400000
                end_checkpoint -= 14400000
                future = executor.submit(requests.post, url_bets, headers= headers, json=data, verify=False)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    response = future.result()
                    print(f"post request has return status code: {response.status_code}")
                    print(count)
                    count += 1
                    response_data = response.json()
                    current_bets = response_data["bets"]
                    all_bets.extend(current_bets)
                except Exception as e:
                    print(f"Error retrieving result from future: {e}")
        if all_bets:
            df = pd.DataFrame(all_bets)
            parquet_filename = f"new_bets_{chunk_index}.parquet"
            df.to_parquet(f"data/{parquet_filename}", engine='pyarrow')       
            print(f"Chunk {chunk_index} saved to {parquet_filename} with {len(df)} bets.")
            chunk_index += 1
            all_bets.clear()
    return 




def modify_bets_currencies():
    df_bets = pd.read_parquet("data/new_bets_0.parquet", engine = "pyarrow")
    df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"], unit='ms')
    df_bets["creation_date"] = df_bets["creation_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_bets["settled_date"] = pd.to_datetime(df_bets["settled_date"], unit='ms')
    df_bets["settled_date"] = df_bets["settled_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_bets["last_update_time"] = pd.to_datetime(df_bets["last_update_time"], unit='ms')
    df_bets["last_update_time"] = df_bets["last_update_time"].dt.strftime('%Y-%m-%d %H:%M:%S')
    print("Start reading exported file")
    df_client_pre = pd.read_parquet("data/exported_clientPT")
    pd.set_option('display.max_columns', 100)
    currency_exchange_rate = {
        'INR': 0.090,
        'EUR': 7.96,
        'TRY': 0.22,
        'USD': 7.79,
        'BRL': 1.27,
        'TZS': 0.0031,
        'CNY': 1.06,
        'PAB': 7.74,
        'GBP': 9.45,
        'HKD': 1.0,
        'BGN': 4.07,
        'XOF': 0.012,
        'VUV': 0.062,
        'ARS': 0.0075,
        'BHD': 20.66,
        'NZD': 4.32,
        'ZMW': 0.28,
        'UGX': 0.0021,
        'SGD': 5.67,
        'TMT': 2.22,
        'TOP': 3.20,
        'TTD': 1.14,
        'UYU': 0.18,
        'WST': 2.78,
        'UAH': 0.18,
        'UZS': 0.0006,
        'AED': 2.12,
        'NOK': 0.68
    }
    df_client_pre["Currency"] = df_client_pre["Currency"].map(currency_exchange_rate).fillna("n/a")
    df_merged = df_bets.merge(
        df_client_pre[['Client ID', 'Currency', "PT", "PT from ME"]],   
        left_on='customer_id',
        right_on='Client ID',
        how='left'  
    )
    cols_to_multiply = [
        "virtual_size",
        "virtual_size_matched",
        "virtual_size_remaining",
        "virtual_profit"
    ]
    for col in cols_to_multiply:
        df_merged[col] = df_merged[col] / df_merged["currency_rate"]
    df_merged.to_parquet("data/df_bets_newday", engine='pyarrow')
    return df_merged
