#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
#from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import MMV_Tools as mmv
plt.rcParams.update(plt.rcParams)

#Now set some default parameters.
plt.rcParams["figure.figsize"] = (15,8) #in inches
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
plt.rcParams['legend.fontsize']=10
plt.rcParams['lines.linewidth']=2

# In[2]:


os.listdir()


# In[3]:
print("####################################")
print("\nIMPORT DATA")

data = pd.read_csv('forecast_carsales.csv')
df = data.copy()
df.reset_index(level = 0, inplace=True)
df.set_index('DATE', inplace=True)
df= df.drop(["index"],axis = 1)


# In[4]:


df.index = pd.to_datetime(df.index)


# In[5]:


df.info()


# In[6]:


df.head(529)


# In[7]:
print("####################################")
print("\nData Subsequenceing")

# 3
# Describe the dataset
# a.
# Plot Dependent(Sales) with time
plt.plot(df.index,df.Sales)
plt.xlabel('Time Step (Monthly)', fontsize=15)
plt.ylabel('Unit Sales in Millions', fontsize=15)
plt.title('Total US Vehicle Sales 1976-2020',fontsize=18)
plt.show()


submean, subvar = mmv.SubSeq(df.Sales)
plt.plot(df.index,submean, label = "Mean")
plt.title(" US Vehicle Sales Mean Sub-Sequence")
plt.ylabel("Unit Sales in Millions")
plt.xlabel("Date")
plt.show()

plt.plot(df.index[2:],subvar, label = "Mean")
plt.title(" US Vehicle Sales Variance Sub-Sequence")
plt.ylabel("Unit Sales in Millions")
plt.xlabel("Date")
plt.show()

submeand, subvard = mmv.SubSeq(np.diff(df.Sales))
plt.plot(df.index[1:],submeand, label = "Mean")
plt.title(" US Vehicle Sales Differenced Mean Sub-Sequence")
plt.ylabel("Unit Sales in Millions")
plt.xlabel("Date")
plt.show()

plt.plot(df.index[3:],subvard, label = "Mean")
plt.title(" US Vehicle Sales Differenced Variance Sub-Sequence")
plt.ylabel("Unit Sales in Millions")
plt.xlabel("Date")
plt.show()

#plt.plot(df.index[3:],subvard, label = "Mean")
#plt.title(" US Vehicle Sales Second - Differenced Variance Sub-Sequence")
#plt.ylabel("Unit Sales in Millions")
#plt.xlabel("Date")
#plt.show()

# In[8]:
print("####################################")
print("\nACF AND STATIONARITY")


mmv.acf_plot(mmv.ACF(df.Sales,len(df)),"Total US Vehicle Sales: 529 Lags")


# In[9]:


mmv.corrmat(df)


# In[10]:


# e
# Data is clean with no missing data values
df.isnull().sum()


# In[11]:


# f
# Split data into training and testing
# Create training and testing sets

train, test = train_test_split(df, test_size = 0.2, shuffle=False)


# In[12]:


# 4
# Stationarity
# Check inital stationarity
mmv.ADF_Cal(df.Sales)


# In[13]:

# Create first order difference of Sales
mmv.ADF_Cal(np.diff(df.Sales))

# In[15]:

mmv.acf_plot(mmv.ACF(np.diff(df.Sales),60),"First Order Difference US Car Sales")
plt.show

# In[16]:

ytrain = train[["Sales"]]
ytest = test[["Sales"]]

# In[17]:

print("####################################")
print("\nDECOMPOSTION")

# Review the differences between additive and multiplicative models
result = seasonal_decompose(ytrain, model ='additive',period = 181)
result.plot()
plt.title("Additive Model")
plt.show()


# In[18]:


result = seasonal_decompose(ytrain, model ='multiplicative',period = 181)
result.plot()
plt.show()


# In[19]:



# In[21]:

print("####################################")
print("\nNAIVE MODEL")

sls = df.Sales
trn = sls[:423]
tst = sls[423:]
drift_yhat, drift_fit = mmv.drift(df.Sales,trn,tst,len(tst))

dresults  = pd.DataFrame(index= ["SEE","MSE"])
dresults["Residual"] = np.array([3398.41, 8.03 ])
dresults["Forecast"] = np.array([1423.20,13.42])

print(dresults)

# In[22]:
print("####################################")
print("\nHOLT WINTERS MODEL SELECT")

ytrain = train[["Sales"]]
ytest = test[["Sales"]]

mmv.holtwinters_modelselect(ytrain,ytest,181)


# In[23]:

print("####################################")
print("\nHOLT WINTERS")

hw_yhat, hw_fit = mmv.holtwinters(ytrain,ytest,"Sales",181,"add","mul")

hwresults  = pd.DataFrame(index= ["SEE","MSE"])
hwresults["Residual"] = np.array([191.95, 0.45 ])
hwresults["Forecast"] = np.array([216.18,2.04])

print(hwresults)





# In[26]:

print("####################################")
print("\nFEATURE SELECTION")

# Multicolinearity
fs = df[["Sales","CPI_UsedV","CapUtil","PayrollNF","UnempRT"]]
fs = fs.copy()
#mmv.corrmat(fs)


# In[27]:


# The only PC to not pass the t-test was Sales-CPI with Payroll controlled
# This makes sense since there is multicolinearity betweent the two
# Will see what OLS regression results show


# In[28]:
print("####################################")
print("\nOLS 1")

yhat, yfit, ytrain, ytest, sef= mmv.linreg(fs,"Sales")


# In[30]:


plt.plot(df.index[:423],yfit, color = "red", label = "Fit")
plt.plot(df.index[:423],ytrain, color = "steelblue", label = "Train")
plt.plot(df.index[423:],ytest, color = "steelblue", label = "Test")
plt.plot(df.index[423:],yhat, color = "orange", label = "Forecast")
plt.title("US Car Sales \n OLS Regression B = 4")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc="best")
plt.show();

# In[32]:
print("####################################")
print("\nOLS 2")

fs1 = fs[["CapUtil","PayrollNF","UnempRT","Sales"]]


# In[33]:


lr_yhat, yfit, ytrain, ytest, sef= mmv.linreg(fs1,"Sales")


# In[34]:


plt.plot(df.index[:423],yfit, color = "red", label = "Fit")
plt.plot(df.index[:423],ytrain, color = "steelblue", label = "Train")
plt.plot(df.index[423:],ytest, color = "steelblue", label = "Test")
plt.plot(df.index[423:],yhat, color = "orange", label = "Forecast")
plt.title("US Car Sales \n OLS Regression B = 3")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc="best")
plt.show();


# In[38]:

print("####################################")
print("\n# LM Parameter Estimation")

# perform first orde difference on data to feed into GPAC
sales = df[["Sales"]]


# In[39]:
sales["Diff"] = np.array(list(np.diff(df.Sales))+ [0])

# In[40]:


sdiff = sales[["Diff"]]
sdm = sdiff.Diff.mean()

sdtrain = sdiff[:423]
sdtest = sdiff[423:]


# In[41]:
print("####################################")
print("\nGPAC")

mmv.GPAC(sdtrain,'Sales Differenced with 423')


# In[ ]:


# LOOKS like an ARMA(4,3)


# In[43]:
print("####################################")
print("\nLM")

num = [1,0,0]
den = [1,0,0,0]
nume, dene, e, var_e, cov_theta = mmv.LM(sdtrain,num,den)


# In[44]:


# only 2 parameters passed CI
# Try another combination (4,2)
num = [1,0,0]
den = [1,0,0,0,0]
nume, dene, e, var_e, cov_theta = mmv.LM(sdtrain,num,den)


# In[ ]:


# In[48]:
print("####################################")
print("\nARMA")

yhat11, yfit11,na11,nb11 = mmv.estARMA(sdtrain,"Diff",sdtest,1,1,0,len(sdtrain)-1,0.01)


# In[49]:

mmv.backtrans(sales,yfit11,yhat11,ytrain,ytest,na11,nb11)


# In[56]:


yhat40,yfit40,na40,nb40 = mmv.estARMA(sdtrain,"Diff",sdtest,4,0,0,len(sdtrain)-1,0.01)


# In[57]:


arma_yhat, arma_yfit = mmv.backtrans(sales,yfit40,yhat40,ytrain,ytest,na40,nb40)


# In[ ]:

print("####################################")
print("\nPREDICTION GRAPH")

#plt.plot(sales.index[:423], ytrain, color = "steelblue", label = "Train")
plt.plot(sales.index[423:], ytest, color = "black", label = "Test")
plt.plot(sales.index[423:], drift_yhat, color = "blue", label = "Naive Drift")
plt.plot(sales.index[423:], hw_yhat, color = "orange", label = "Holt-Winters")
plt.plot(sales.index[423:], lr_yhat, color = "red", label = "OLS")
plt.plot(sales.index[423:], arma_yhat, color = "green", label = "ARMA(4,0)")
plt.title("All Model Predictions for US Vehicle Sales")
plt.ylabel("Unit Sales in Millions")
plt.xlabel("Date")
plt.legend(loc = "upper left")
plt.show()


print("####################################")
print("\nEND OF PROJECT")