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


Feature engineering

| Engineered Features             |
|---------------------------------|
| Bet Frequency per day           |
| Bet Frequency per week          |
| Bet Frequency per month         |
| Total lifetime Bets             |
| Average pnl                     |
| Frequency encoding competitions |

