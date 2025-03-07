# Fraudulent bets detection model for bet exchange

# Project Overview

As a data scientist for my betting company, my first machine learning model I am building is a Fraudulent Bet detection model.
As online sports betting has become more popular, betting companies in the UK have created new ways of betting - Betting Exchange.
Similar to stock markets, betting exchange is a marketplace where customers can bet against each other in the exchange, 
and they are able to set their own price as a bookmaker with a few differences. Gamblers can buy or sell the outcome.
This created a new way of betting and entertainment for the customers, and bet exchange operators generate revenue by 
charging a small commission on winning bets.

The problem comes in now, like other finacial transactions, we are seeing increasingly more fradulent bets placed in the
betting exchange by betting against themself, to achieve fund passing, and some trying to make profits. 
See more at https://arbusers.com/fund-passing-at-betfair-t6198/

Bookmaker companies have been trying to catch fraudulent bets mannually for years, however it is highly insufficient and prone to error,
and these fradulent bets clients are likely to be using bots to achieve their purpose.

My job for the company and my first real world machine learning problem is to build a working machine learning model to detect
these fradulent bets efficiently and accurately.

# Data Science problem and my approach

I am using a binary classfication model where I classfy fradulent bets as 1 and non fradulent bets as 0.
Currently my model is still a beta model and I am only using XGBoost Classifier for now as I am still in the data engineering progress.

# Data collection

I have 3 years of bet data from the betting exchange, which is around 70 million rows of bets. As the company has been mannually finding fraudulent bets, we have around 20000 confirmed
fradulent bets for me to use in my training and test model.

I am dividing my dataset to two parts, where first 2 years of data is used for the training and testing sets of the model. The more recent
year of data is used as the validation set for final checking, before I deploy it to detect frauds from daily new bets.

# Feature Selections

There are currently around 47 different features I am still testing and comparing with, where most features I engineered myself.
Here is a table of all the data I engineered with the original raw data from our back end database, with most of them being quite important for
our model.

| Engineered Features                | Definition                                                                                        |
|------------------------------------|---------------------------------------------------------------------------------------------------|
| price_ratio                        | Price of the bet / Average price of the bet's selection                                           |
| competition_counts                 | The number of times of the competition of this bet has appeared in all the bets.                  |
| total_pnl                          | The average career PnL of this customer                                                           |
| lifetime_bets                      | The number of bets the customer has placed in the past with this account                          |
| pnl_7days                          | The total PnL of the customer in the last 7 days                                                  |
| num_of_client_bets_before_bet      | The number of bets that the customer has placed with this account in the past                     |
| total_client_profit_before_bet     | The total profit of the customer in this account before this bet was placed                       |
| total_client_roi                   | The history ROI of the account of this bet                                                        |
| registration_time                  | The creation date of the first bet that the customer made, in "ms"                                |
| time_since_first_bet               | The time between this bet and the first bet that the account has placed, in "ms"                  |
| selection_encoded                  | Encoded the string column selection name into a numerical column                                  |
| event_name_encoded                 | Encoded the string column event name into a numerical column                                      |
| customerid_encoded                 | Encoded the string column customer id into a numerical column                                     |
| betplace_matchstart_timedifference | The time difference between the start time of the match versus the place time of the bet, in "ms" |
| total_selection_bets_num           | The total number of bets placed for this specific bet selection                                   |
| total_selection_stakes             | The total stakes matched for this specific bet selection                                          |
| num_of_selection_bets_before       | The number of bets placed for this specific bet selection before this bet was placed              |
| 10mins_avg_price                   | The mean price of this bet selection over the last 10 minutes before this bet was placed          |
| price_range                        | The range of the price of this specific bet selection                                             |

