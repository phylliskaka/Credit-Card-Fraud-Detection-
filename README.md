# Credit-Card-Fraud-Detection-
Predicting the transaction is fraud or not.      
## Introduction 
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.      

## Dataset
Dataset name: Synthetic Financial Datasets For Fraud Detection    
dataset link: https://www.kaggle.com/ntnu-testimon/paysim1/downloads/paysim1.zip/2 .  

There is a lack of public available datasets on financial services and specially in the emerging mobile money transactions domain. Part of the problem is the intrinsically private nature of financial transactions, that leads to no publicly available datasets.     

We present a synthetic dataset generated using the simulator called PaySim as an approach to such a problem. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods.   

Dataset is size of (6362620, 11), 11 columns name are described following:   
1. step: Maps a unit of time in the real world. In this case 1 step is 1 hour of time.   
2. type: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.    
3. amount: amount of the transaction in local currency.  
4. nameOrig: customer who started the transaction.    
5. oldbalanceOrg: initial balance before the transaction.  
6. newbalanceOrig: customer's balance after the transaction.   
7. nameDest: recipient ID of the transaction.    
8. oldbalanceDest: initial recipient balance before the transaction.    
9. newbalanceDest: recipient's balance after the transaction.    
10. isFraud: identifies a fraudulent transaction (1) and non fraudulent (0).  (Target)   
11. isFlaggedFraud: flags illegal attempts to transfer more than 200.000 in a single transaction.   

## Data Preprocessing
1. Data Cleaning: check and fill the null value in dataset.    
2. Feature selection: drop the features `nameOrig`, `nameDest`.
3. Feature Encoding: transfer category feature `type` to numeric feature.  
4. PCA: reduce data dimenstionality.   
![alt text]()
5. Random Undersampling

## Prediction 
Model: Logistic Regression.   

## Result
Achieving 87% percent testing accuracy.   
