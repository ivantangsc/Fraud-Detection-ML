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

logging.getLogger().setLevel(logging.INFO)



#%%

APIKEY="dataminingEFkRF4rfJSKe23"
USER_NAME="API_sub"
PASSWORD="Passw0rd!"

url="https://agents.orbittest.com/api/sa/sa_login"
url_account_tree="https://agents.orbittest.com/sa/get_account_tree"
url_pro_account_tree = "https://external.orbitexch.com/api/sa/get_account_tree"


APIKEY_pro="dataminingprAmfh843FerPOa23"
USER_NAME_pro="TungTungCOMP"
PASSWORD_pro="w3st0ngp870kl$"

url_pro = "https://external.orbitexch.com/api/sa/sa_login"
url_pro_account_tree = "https://external.orbitexch.com/api/sa/get_account_tree"
url_pro_account_list = "https://external.orbitexch.com/api/sa/get_ma_accounts_list"
url_pro_biab = "https://external.orbitexch.com/api/get_biab_accounts_for_ma"
url_bets = "https://external.orbitexch.com/api/get_bets_external"

headers = {
    "X-API_KEY": APIKEY_pro,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.5",
    "Content-type": "Application/JSON",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "Referer": "https://external.orbitexch.com/",
    "Cookie": "date_lang=en"
}
data = {
    "username": USER_NAME_pro,
    "password": PASSWORD_pro
}



response = requests.post(url_pro, headers=headers, json=data, verify=False)
response.text
token = response.json()["token"]

current_time = datetime.datetime.now()
start_of_today = datetime.datetime(current_time.year, current_time.month, current_time.day)
start_of_yesterday = start_of_today - datetime.timedelta(days=1)
end_checkpoint = int(time.mktime(start_of_yesterday.timetuple())) * 1000

print(end_checkpoint)
desired_date = datetime.datetime(2025, 2, 18, 20, 0)
desired_checkpoint = int(time.mktime(desired_date.timetuple())) * 1000
print(desired_checkpoint)
day = 6
chunk_index = 0

# %%

headers = {
    "Authorization": f"Bearer {token}",
    "X-API_KEY": APIKEY_pro,
    "Content-type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.5",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "Referer": "https://external.orbitexch.com/",
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
                "state": [4,5,6,7,8,9,10]

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
        df.to_parquet(parquet_filename, engine='pyarrow') 
        
        print(f"Chunk {chunk_index} saved to {parquet_filename} with {len(df)} bets.")
        chunk_index += 1

        all_bets.clear()

# %%

df_bets = pd.read_parquet("new_bets_0.parquet", engine = "pyarrow")

df_bets["creation_date"] = pd.to_datetime(df_bets["creation_date"], unit='ms')
df_bets["creation_date"] = df_bets["creation_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
df_bets["settled_date"] = pd.to_datetime(df_bets["settled_date"], unit='ms')
df_bets["settled_date"] = df_bets["settled_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
df_bets["last_update_time"] = pd.to_datetime(df_bets["last_update_time"], unit='ms')
df_bets["last_update_time"] = df_bets["last_update_time"].dt.strftime('%Y-%m-%d %H:%M:%S')



print("Start reading exported file")

df_client1 = pd.read_excel('exported_file.xlsx', sheet_name = 'Accounts 1')
print("Process reading sheet 1 finished")
df_client2 = pd.read_excel('exported_file.xlsx', sheet_name = 'Accounts 2')
print("Process reading sheet 2 finished")
df_client3 = pd.read_excel('exported_file.xlsx', sheet_name = 'Accounts 3')

df_client_pre = pd.concat([df_client1, df_client2, df_client3], ignore_index = True)

print("Process reading 3 sheets finished")


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
    df_merged[col] = df_merged[col] * df_merged["Currency"]



df_merged.to_parquet("df_bets_newday", engine='pyarrow') 


