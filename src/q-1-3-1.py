#!/usr/bin/env python
# coding: utf-8

# ### Q-1-3-1
# 
# Implement a model using linear regression to predict the probablity of getting the admit.

# In[11]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys


# loading data from csv to dataframe

# In[12]:


df = pd.read_csv("../input_data/AdmissionDataset/data.csv")
X  = df.drop(['Serial No.', 'Chance of Admit '],axis='columns')
Y  = df['Chance of Admit ']
X = (X - X.mean())/X.std()
# X  = pd.DataFrame( preprocessing.scale(X), columns = [ i for i in X ])
# X  = pd.concat([  X, df['Research'] ], axis='columns')
# X.head()


# append column of ONES at 0th index. 

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)

ones = pd.DataFrame(1, index=np.arange(X_train1.shape[0]), columns=["a"])

X_train1 = pd.concat( [ones, X_train1], axis='columns')

X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape((X_train1.shape[0],1))


# method to calculate values of theta using gradient descent

# In[14]:


theta = np.zeros([1,8])
alpha = 0.01
iters = 1000

def grad_desc(X,y,theta,iters,alpha):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * ( np.matmul(X , theta.T) - y), axis=0)
    return theta

g = grad_desc(X_train1,Y_train1,theta,iters,alpha)


# predict function 

# In[15]:


thetalist = g[0]
def predict(X_test):
    y_pred= list()
    for index,row in X_test.iterrows():
        row=list(row)
        y1=0
        for i in range(1,8):
            y1=y1+thetalist[i]*row[i-1]
        y1=y1+thetalist[0]
        y_pred.append(y1)
    return y_pred
y_pred_mymodel = predict(X_test)

y_pred_mymodel
# In[16]:


r2_score(list(Y_test),y_pred_mymodel)


# inbuilt sckit-learn Linear Regression.

# In[17]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred_system = regressor.predict(X_test) 


# printing theta for both lists.

# In[18]:


print [regressor.intercept_ ] + list(regressor.coef_ )
print(thetalist)

print y_pred_system
# In[19]:


print r2_score(Y_test,y_pred_system)


# In[20]:


test_file = sys.argv[1]
df_test = pd.read_csv(test_file)
test_pred = predict( df_test )
print test_pred

