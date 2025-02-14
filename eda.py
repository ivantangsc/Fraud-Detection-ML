# %% EDA 

import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
import time
import json
import urllib3
import seaborn as sns
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

test_sample = pd.read_parquet('test_sample', engine = 'pyarrow')
test_sample

# %%
fraud_bet = test_sample[test_sample['fraud'] == 1]
non_fraud_bet = test_sample[test_sample['fraud'] == 0]

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,2)
plt.bar(non_fraud_bet['event_type_name'].value_counts().head(10).index, non_fraud_bet['event_type_name'].value_counts().head(10).values, edgecolor = 'black')
plt.title('Distribution of event types for regular bets')
plt.xlabel('Event types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,1)
plt.bar(fraud_bet['event_type_name'].value_counts().head(10).index, fraud_bet['event_type_name'].value_counts().head(10).values, color = 'indianred', edgecolor = 'black')
plt.title('Distribution of event types for fraud bets we find')
plt.xlabel('Event types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)


plt.tight_layout()
plt.show()


# %%

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,2)
plt.bar(non_fraud_bet['market_name'].value_counts().head(10).index, non_fraud_bet['market_name'].value_counts().head(10).values, edgecolor = 'black')
plt.title('Distribution of market types for regular bets')
plt.xlabel('Market types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,1)
plt.bar(fraud_bet['market_name'].value_counts().head(10).index, fraud_bet['market_name'].value_counts().head(10).values, color = 'indianred', edgecolor = 'black')
plt.title('Distribution of market types for fraud bets we find')
plt.xlabel('Market types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)


plt.tight_layout()
plt.show()

# %%

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,2)
plt.bar(non_fraud_bet['competition_name'].value_counts().head(10).index, non_fraud_bet['competition_name'].value_counts().head(10).values, edgecolor = 'black')
plt.title('Distribution of competition types for regular bets')
plt.xlabel('Competition types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,1)
plt.bar(fraud_bet['competition_name'].value_counts().head(10).index, fraud_bet['competition_name'].value_counts().head(10).values, color = 'indianred', edgecolor = 'black')
plt.title('Distribution of competition types for fraud bets we find')
plt.xlabel('Competition types')
plt.ylabel('Frequency')
plt.xticks(rotation = 45)


plt.tight_layout()
plt.show()
# %%

plt.figure(figsize = (12 , 16))
plt.subplot(2,1,2)
plt.hist(non_fraud_bet['price'], bins = 50, edgecolor = 'black')
plt.title('Distribution of price for regular bets')
plt.ylabel('Price')


plt.figure(figsize = (12 , 16))
plt.subplot(2,1,2)
plt.hist(fraud_bet['price'], bins = 50, color = 'indianred', edgecolor = 'black')
plt.title('Distribution of price for fraud bets we find')
plt.ylabel('Price')


plt.tight_layout()
plt.show()



# %%

plt.figure(figsize=(12, 6))

plt.bar_label(test_sample.groupby('fraud')['lifetime_bets'].mean().plot(kind='bar', color=['steelblue', 'indianred'], edgecolor='black').containers[0])

plt.title("Average Lifetime bets of an Account", fontsize=16)
plt.xlabel("Fraud Type (0 = Non-Fraud, 1 = Fraud)", fontsize=14)
plt.ylabel("Average Lifetime", fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()







# %% 
# Bivarite Analysis

from scipy.stats import pearsonr

numerical_sample = test_sample.select_dtypes(exclude = ['object', "category"])

correlation_matrix = numerical_sample.corr()
fraud_correlations = correlation_matrix['fraud']
sorted_correlations = fraud_correlations.sort_values(ascending=False)

print(sorted_correlations)