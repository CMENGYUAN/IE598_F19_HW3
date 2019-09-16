#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv('file:///Users/mengyuanchen/Desktop/HY_Universe_corporate bond.csv')
df.head()
#code is original from Datacamp and Machine Learning in Python 
#getting help from Yuzheng Nan


# In[2]:


#Listing 2.1 
summary = df.describe()
print(df.head())
print(summary)
print('num of row is '+ str(df.shape[0]))
print('num of col is '+ str(df.shape[1]))


# In[3]:


#histogram of days of trade
_=plt.hist(df['n_days_trade'])
_=plt.xlabel('n days trade')
_=plt.ylabel('number of bonds')
plt.show()


# In[4]:


#creating ecdf
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

x_liq,y_liq=ecdf(df['n_days_trade'])
_=plt.plot(x_liq, y_liq, marker='.', linestyle='none')
_=plt.xlabel('n days trade')
_=plt.ylabel('ECDF')
plt.show()


# In[12]:


#Bee Swarmplot 
Electric=df.loc[(df['Industry']=='Electric')]
_=sns.swarmplot(y='n_days_trade',data=Electric)
plt.title('Swarmplot of n_days_trade of electric industry')
plt.show()


# In[13]:


_=sns.boxplot(y='n_days_trade',data=Electric)
plt.title('Boxplot of n_days_trade of electric industry')
plt.show()


# In[18]:


#Scatterplot between volume and day of trade
_=plt.plot(df['volume_trades'],df['n_days_trade'],marker='.',linestyle='none')
plt.show()


# In[20]:


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient
r=pearson_r(df['volume_trades'],df['n_days_trade'])

# Print the result
print('Pearson correlation coefficient of volume_trades and n_days_tarde:',r,'\n')


# In[ ]:


print("My name is {Mengyuan Chen}")
print("My NetID is: {mchen100}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

