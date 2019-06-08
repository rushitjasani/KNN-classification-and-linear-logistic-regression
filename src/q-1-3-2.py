#!/usr/bin/env python
# coding: utf-8

# ### Q-1-3-2
# 
# Compare the performance of Mean square error loss function vs Mean Absolute error function vs Mean absolute percentage error function and explain the reasons for the observed behaviour.

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[2]:


df = pd.read_csv("../input_data/AdmissionDataset/data.csv")
X  = df.drop(['Serial No.', 'Chance of Admit '],axis='columns')
Y  = df['Chance of Admit ']
X = (X - X.mean())/X.std()


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train1 = X_train.reset_index(drop=True)
Y_train1 = Y_train.reset_index(drop=True)

ones = pd.DataFrame(1, index=np.arange(X_train1.shape[0]), columns=["a"])

X_train1 = pd.concat( [ones, X_train1], axis='columns')

X_train1 = np.array(X_train1)
Y_train1 = np.array(Y_train1).reshape((X_train1.shape[0],1))


# In[4]:


theta = np.zeros([1,8])
alpha = 0.01
iters = 1000

def grad_desc(X,y,theta,iters,alpha):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * ( np.matmul(X , theta.T) - y), axis=0)
    return theta

g = grad_desc(X_train1,Y_train1,theta,iters,alpha)


# In[5]:


thetalist=g[0]

y_pred_mymodel=[]
for index,row in X_test.iterrows():
    row=list(row)
    y1=0
    for i in range(1,8):
        y1=y1+thetalist[i]*row[i-1]
    y1=y1+thetalist[0]
    y_pred_mymodel.append(y1)

y_pred_mymodel
# In[6]:


r2_score(list(Y_test),y_pred_mymodel)


# In[ ]:





# In[7]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred_sys = regressor.predict(X_test) 


# In[8]:


print(regressor.coef_)
print(thetalist)
print(regressor.intercept_)


# functions to calculate all 3 types of errors.

# In[9]:


def mean_absolute_percentage_error(y_test, y_pred):
    return 100.0 * np.mean( np.abs(y_test - y_pred) / y_test )

def mean_absolute_error(y_test, y_pred):
    return np.mean( np.abs(y_test - y_pred))

def mean_squared_error( y_test, y_pred ):
    return np.mean((y_test - y_pred)**2 )


# In[10]:


print mean_absolute_error(Y_test,y_pred_sys)
print mean_squared_error(Y_test,y_pred_sys)
print mean_absolute_percentage_error(Y_test,y_pred_sys)


# In[11]:


print mean_absolute_error(Y_test,y_pred_mymodel)
print mean_squared_error(Y_test,y_pred_mymodel)
print mean_absolute_percentage_error(Y_test,y_pred_mymodel)


# ### Observations
# 
# * MSE has the benefit of penalizing large errors more so can be more appropriate in some cases, for example, if being off by 10 is more than twice as bad as being off by 5. But if being off by 10 is just twice as bad as being off by 5, then MAE is more appropriate.MAE gives less weight to outliers, which is not sensitive to outliers.
# 
# * Mean Absolute Percentage Error is Similar to MAE, but normalized by true observation. Downside is when true obs is zero, this metric will be problematic.
# 
# 

# In[ ]:




