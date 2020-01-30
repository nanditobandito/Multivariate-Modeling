import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import os

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" %result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print('\t%s: %3f' % (key,value))

print(os.listdir())

# 1 Load the time series data
df = pd.read_csv("AirPassangers.csv")

# 2 Data covers 1949-1960
# 3 Display the first 5 rows of the dat

print(df.head(5))

# 4 Plot the entire dataset
# Is there a trend, seasonality, cyclic
# Add label to horizonatal and vertical axis (month vs sales)
# # Add Title "Air passangers Dataset without differencing"
# Add a legend

ax = df.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers Dataset without Differencing',fontsize=18)
plt.show()

# 5 is the data stationary or non-stationary?
# Justify answer.
# Calculate the average over the entire data set and show the average plot

print("Summary Stats:",df.describe())
sales_means = []
for i in range(0,144):
    sales_means.append(df.head(i).mean())
# Convert list into dataframe
df_sales_mean = pd.DataFrame(sales_means)

# Plot average data
ax = df_sales_mean.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passengers Dataset Cumulative Average without Differencing',fontsize=18)
plt.show()

# Since there is a clear increasing trend in the mean over time, then this means the data is non-stationary
# Perform ADF test to objectively show that data is either sationary or non-stationary

ADF_Cal(df["#Passengers"])

# 6 If the data is non-stationary then write a python code that will detrend it by the 1st order difference transformation.
# Plot the detrended data set.

# Create a new list with just passenger information
sales_means1 = []
for i in range(0,145):
    sales_means1.append(df["#Passengers"].head(i).mean())

# Convert passenger list into a numpy array
sm_diff_arr = np.array(sales_means1)
# apply first order differncing on the passenger array
sm_diff = np.diff(sm_diff_arr)
print("Input array  : ", sm_diff_arr)
print("First order difference  : ", sm_diff)

# Convert first order differencing array into a pandas dataframe
df_sm_diff = pd.DataFrame(sm_diff)
# Rename columns
df_sm_diff = df_sm_diff.rename(columns={0: "#Passengers"})
# drop first observation since it is missing due to differencing
df_sm_diff = df_sm_diff.drop([0])

# Plot the first order difference of passenger data
ax = df_sm_diff.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passengers Dataset with Differencing',fontsize=18)
plt.show()

# 7 Is the detrended dataset stationary? Justify Answer.

# Since there is no clear increasing trend in the mean over time, then this means the data is stationary.
# Perform ADF test to objectively show that data is either sationary or non-stationary

ADF_Cal(df_sm_diff["#Passengers"])

# Calculate average over the entire dataet and show the average plot
# Change range from 1,145 since we dropped the first obsevation using differencing
sales_means_diff = []
for i in range(1,145):
    sales_means_diff.append(df_sm_diff["#Passengers"].head(i).mean())

df_sales_cdm = pd.DataFrame(sales_means_diff)
# Rename columns
df_sales_cdm = df_sales_cdm.rename(columns={0: "#Passengers"})

# Plot the first order difference overall mean of passenger data
ax = df_sales_cdm.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passengers Dataset Cumulative Average with Differencing',fontsize=18)
plt.show()

# 8 Using the logarithimc transformation mehtod, and differencing method, detrend the data.
# Use numpy library and convert air passenger numbers inot logarthimic scale
# Take the difference between two adjacent observations
# Plot results

# Use Second Order Differencing
# Convert passenger list into a numpy array
first_diff = np.array(df["#Passengers"])
# apply first order differncing on the passenger array
first_diff = np.diff(first_diff)
second_diff = np.diff(first_diff)

df_second_diff = pd.DataFrame(second_diff)
# Rename columns
df_second_diff = df_second_diff.rename(columns={0: "Passengers"})
df_second_diff

# Plot the Second difference of passenger data
ax = df_second_diff.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower left', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers Second Order Difference',fontsize=18)
plt.show()

ADF_Cal(df_second_diff["Passengers"])


# Create new column taking the log(base 10) of passenger data
df["LogPassengers"] = np.log(df['#Passengers'])

# Plot the logarithmic transformation
# Logoarithmic transformation removes varrying variance
ax = df["LogPassengers"].plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers Logarithmic Transformation',fontsize=18)

# Convert passenger list into a numpy array
log_sm_arr = np.array(df["LogPassengers"])
# apply first order differncing on the passenger array
log_sm_diff = np.diff(log_sm_arr)

df_log_diff = pd.DataFrame(log_sm_diff)
# Rename columns
df_log_diff = df_log_diff.rename(columns={0: "LogPassengers"})

# Plot the logarithmic difference of passenger data
ax = df_log_diff.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers First Order Logarithmic Difference',fontsize=18)
plt.show()

# 9 is the transformed Data now stationary?
# Justify answer.

# Calculate average over the entire dataset and show average plot.
log_means = []
for i in range(0,145):
    log_means.append(df_log_diff["LogPassengers"].head(i).mean())

df_log_means = pd.DataFrame(log_means)
# Rename columns
df_log_means = df_log_means.rename(columns={0: "LogPassengers"})

# Plot Log Means
ax = df_log_means.plot(kind='line',figsize=(10,8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers Logarthimic Transformation Cumalitive Mean',fontsize=18)
plt.show()

# Visually, since there is no clear increasing trend in the mean over time, then this means the data is stationary.
# Perform ADF test to objectively show that data is either sationary or non-stationary

ADF_Cal(df_log_diff["LogPassengers"])

# Calcualte variance and show transformation converts the non-stationary data into stationary
log_var = []
for i in range(1, 145):
    log_var.append(df_log_diff["LogPassengers"].head(i).var())

df_log_var = pd.DataFrame(log_var)
# Rename columns
df_log_var = df_log_var.rename(columns={0: "LogPassengers"})
df_log_var = df_log_var.drop([0])

# Plot Log Means
ax = df_log_var.plot(kind='line', figsize=(10, 8), fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax.set_xlabel('Months', fontsize=15)
ax.set_ylabel('Sales', fontsize=15)
ax.set_title('Air Passangers Logarthimic and Differencing Cumulative Variance', fontsize=18)
plt.show()

# Perform ADF test on transforemd cumalitive variance
ADF_Cal(df_log_var["LogPassengers"])

