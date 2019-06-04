#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:04 2019

@author: wzhan
"""

# Importing package 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler

#%%
# Read dataset 
ROOT_PATH = os.getcwd()
path = os.path.join(ROOT_PATH, 'PS_20174392719_1491204439457_log.csv')
df = pd.read_csv(path)

#%% Understand dataset(feature name, data type, misssing value and ect.)
print(df.columns)
print(df.dtypes)
print(df.head())
df.isnull().sum()

#%%
# Drop the feature that is not helping prediction 
df.drop(['nameOrig', 'nameDest'], axis = 1, inplace = True)

# Transfer category feature to numberic feature
# Count the frequency of columns 'type'
df['type'].value_counts()
# It has CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT 
df_num = pd.get_dummies(df)

#%%
# Plot the correlation between features and target 
plt.figure(figsize = (12, 12))
sns.heatmap(df_num.corr(), annot = True, cmap = plt.cm.Reds)
plt.show()

#%%
# Perform PCA 
## Standardize the data
### Features name that exclude isFraud
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
            'newbalanceDest', 'isFlaggedFraud', 'type_CASH_IN', 'type_CASH_OUT', 
            'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# Seperate the feature columns and target colums 
X_features = df_num.loc[:, features]
y = df_num['isFraud']
x = StandardScaler().fit_transform(X_features)

# Fit PCA on X_features 
from sklearn.decomposition import PCA 
pca = PCA(n_components = len(features))

transformed_data = pca.fit_transform(x)
print(transformed_data.shape)
print(pca.explained_variance_ratio_*100)
print(pca.explained_variance_)

# Plot the principal components in pie chart, and only keep the components that 
# contains 95% information of original dataset.
threshold = 0.95
for_test = 0
order = 0 
for index, ratio in enumerate(pca.explained_variance_ratio_):
    if threshold > for_test:
        for_test += ratio 
    else:
        order = index + 1 
        break 

print(pca.explained_variance_ratio_[:order].sum())
com_col = ['com'+ str(i+1) for i in range(order)]
com_col.append('others')
com_value = [i for i in pca.explained_variance_ratio_[:order]]
com_value.append(1-pca.explained_variance_ratio_[:order].sum())
plt.figure(figsize=[4,4])
plt.pie(x = com_value, labels = com_col)
plt.title('Principal components')
plt.show()

import gc
del transformed_data, threshold, for_test
gc.collect()

del X_features 
gc.collect()
#%%
# Reduce data dimensionality from 13 to 9, feature columns become V1, V2,....V9
pca_new = PCA(n_components = 9)
transformed_data = pca_new.fit_transform(x)
X = pd.DataFrame(data = transformed_data, 
                 columns = ['V' + str(i) for i in range(1, 10)])
df = pd.concat([X, y], axis = 1)

#%%
# Helper function for evaluate prediction result. 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

def clf_evaluate(y_pred, y_test):
    '''
    The function for evaluate the model. 
    
    Parameters: 
        y_pred(np array): the prediction from classifier 
        y_test(np array): the groundtruth of label 
    
        
    '''
    confusion = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('Confusion Matrix')
    print(confusion)
    print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1: {3:.4f}'.format(acc, prec, rec, f1))


#%%
# Predicting using Logistic Regression claassifier    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## Undersampling dataset 
df.info()
df_copy = df.copy()
count_class_0, count_class_1 = df_copy.isFraud.value_counts()

df_copy_0 = df_copy[df_copy['isFraud'] == 0]
df_copy_1 = df_copy[df_copy['isFraud'] == 1]

df_copy_0_under = df_copy_0.sample(count_class_1)
df_under = pd.concat([df_copy_0_under, df_copy_1], axis = 0)

X_features = df_under.drop(['isFraud'], axis = 1)
y_target = df_under.isFraud

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                                                    test_size = 0.3, random_state =42,
                                                    stratify = y_target)
clf =  LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf_evaluate(y_pred, y_test)



