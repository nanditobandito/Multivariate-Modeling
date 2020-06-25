#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pp
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[26]:


def ACF(data,lags):
    # convert input data into a numpy array
    data = np.array(data)
    # acf will store the autocorreltion coefficent at each lag interval
    # the first datapoint is always 1.0 since anything correlated with itsself is = 1
    acf = [1.0]
    # calculate the mean for the entire dataset
    y_bar = data.mean()
    print("The mean of this dataset is: ",y_bar)
    # subtract the mean from each observation
    yy_bar = data - y_bar
    # clacualte the total variance for the data set
    total_variance = sum(np.square(yy_bar))
    #print("The total variance for this dataset is: ", total_variance)
    # perform a forloop over the dataset with the desired number of lags
    # range is 1,lags b/c the first iteration calcualtes T1
    for i in range(1,lags):
        # first nparray is removing the last element each iteration
        yy_bar_bottom = yy_bar[:-i]
        # second nparray removes the first element each interation
        yy_bar_top = yy_bar[i:]
        # take the sum of of the product of each nparray each iteration
        yy = sum(yy_bar_top * yy_bar_bottom)
        # divide the sum by total variance and append to resulting acf list
        acf.append(yy/total_variance)
    return acf


# In[27]:


def acf_plot(y,a):
    #y = y.tolist()
    y_rev = y[::-1]
    y_rev.extend(y[1:])
    print(len(y_rev))
    lb = -(math.floor(len(y_rev)/2))
    hb = -(lb-1)
    x = np.array(list(range(lb,hb)))
    figure = plt.stem(x,y_rev,use_line_collection=True)
    plt.xlabel('Lag', fontsize=15)
    plt.ylabel('AC Coefficent', fontsize=15)
    plt.title('ACF with {} samples'.format(a),fontsize=18)
    plt.show()

    #return y_rev


# In[28]:


def GPAC(y,a):
    acf = ACF(y, 30)
    acf_plot(ACF(y, 15),a)
    # construct den matrix
    den = np.zeros([14, 7])
    for j in range(0, 14):
        for k in range(1, 8):
            den[j][k - 1] = acf[abs(j - k + 1)]

    # GPAC matrix
    phikk = np.zeros([7, 7])
    for j in range(0, 7):
        for k in range(0, 7):
            if k == 0:
                d = den[j][k]
                n = den[j + 1][k]
                phi = n / d
                if d < 0.001:
                    phi = 0
                phikk[j][k] = phi
            else:
                d = den[j:j + k + 1, :k + 1]
                # capture the den info for num
                n1 = den[j:j + k + 1, :k]
                # create j+k column
                n2 = np.array(acf[j + 1:j + k + 2])
                num = np.concatenate([n1, n2], axis=1)
                phi = (np.linalg.det(num)) / (np.linalg.det(d))
                dt = (np.linalg.det(d))
                if dt < 0.001:
                    phi = 0
                phikk[j][k] = phi

    # Plot table
    sns.heatmap(phikk, annot=True, vmax=.1, vmin=-.1)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.title("GPAC with {} samples".format(a))
    plt.xticks(np.arange(0.5, len(phikk), 1), np.arange(1, 8, 1))
    #plt.show()


# In[29]:


def ARMA(N,AR,MA,na,nb):
    # Setup the error signal with N # of samples
    mean = 0
    std = 1
    np.random.seed(42)
    e = std * np.random.randn(N) + mean
    
    print("\nARMA with AR order {0} and MA order {1}".format(AR,MA))
    num = nb
    print("MA",num)
    den = na
    print("AR",den)
    
    system = (num,den,1)
    x,y = signal.dlsim(system,e)
    plt.plot(y)
    plt.show()
    return y


# In[30]:


def ARMA_gen():
    a = int(input("\nEnter the numbers of samples :"))
    b = int(input("\nEnter the order # of the AR process :"))
    c = int(input("\nEnter the order # of the MA process :"))
    print("\n The program will ask you to enter each parameter indivdually")
    na = np.zeros(b)
    for i in range(len(na)):
        na[i]= (float(input("\nEnter paramter {0} of AR({1}):".format(i+1,b))))
    if b < c:
        x = np.zeros(c-b)
        na = np.array(list(na) + list(x))
    d = np.array([1] + list(na))
    
    nb = np.zeros(c)
    for  i in range(len(nb)):
        nb[i] = (float(input("\nEnter parameter {0} of MA({1}):".format(i+1,c))))
    if c < b:
        z = np.zeros(b-c)
        nb = np.array(list(nb) + list(z))
    e = np.array([1] + list(nb))
    
    results = ARMA(a,b,c,d,e)
    
    GPAC(results,a)
    
    return results


# In[101]:


def step0(num_true,den_true):
    print("\nStep 0:")
    # create the true theta parameters vector based on na,nb parameters
    if len(den_true) == 1:
        theta_true = np.vstack((0,np.vstack(num_true[1:])))
    elif len(num_true) == 1:
        theta_true = np.vstack((np.vstack(den_true[1:]),0))
    else:
        theta_true = np.vstack((np.vstack(den_true[1:]),np.vstack(num_true[1:])))     
    theta_true = [aa for bb in theta_true for aa in bb]
    # Initialize theta vector to zero based on the # of paremeters
    # Revmove the 1 at the begining of num and den since they are for dlsim package and NOT ture parameters
    nb = len(num_true) -1
    na = len(den_true) -1
    # number of parameters
    NP = na + nb
    # intialize parameters to 0 for the length of na + nb
    theta = np.zeros(NP)
    theta = theta.reshape(len(theta),1)
    print("Theta true:", theta_true)
    print("# na parm:",na)
    print("# nb parm:",nb)
    print("Theta shape:",theta.shape)   
    
    return theta_true,theta,na,nb


# In[251]:


# Function that takes in theta parameters and y
# Calculates white noise of function y
def calsim(y,theta,na,nb):
    print("\nCalsim:")
    num = [1] + list(theta[na:nb+1].flatten())
    print("Num:",num)
    den = [1] + list(theta[:na].flatten())
    print("Den:",den)
    system = (den,num,1)
    _,e = signal.dlsim(system,y)
    # Change e from tuple to an array
    enew = np.zeros(len(e))
    for i in range(len(e)):
        enew[i] = e[i][0]
        
    enew = enew.reshape(len(e),1)
    print("Shape of e: ",enew.shape)
    print("Length of e: ",len(enew))

    return enew


# In[252]:


# Function that calcuates the SSE old
# Creates A and g paramters
def step1(y,theta,delta,na,nb):
    print("Step 1:")
    e = calsim(y,theta,na,nb)
    SSEo = float(np.dot(e.T,e))
    print("\nSSE old:",SSEo)
    
    X = np.zeros([na+nb,len(e)])
    for i in range(na+nb):
        print("\n theta {} + delta:".format(i+1))
        theta2 = theta.copy()
        theta2[i] = theta[i] + delta
        e2 = calsim(y,theta2,na,nb)
        X[i] = ((e - e2)/delta).flatten()
        print("Shape of X:",X.shape)
        
    # Need to transpose the X matrix for the proper shapes    
    X = X.T        
    print("\nShape of X:",X.shape)
    A = X.T @ X
    print("\nMatrix A:",A)
    print("Shape of A:", A.shape)
    g = X.T @ e
    print("\ng:",g)
    print("Shape of g:", g.shape)
    
    return A,g,SSEo


# In[253]:


# Step 2 update theta
def step2(y,theta,A,g,mu,na,nb):
    print("\nStep 2:")
    # Create Identity matrix with dimensions NP x NP
    I = np.identity(na+nb)
    print("\nmu:",mu)
    muI = np.dot(mu,I)
    # Calculate change in theta
    dtheta = np.dot(np.linalg.inv(A + muI),g)
    print("\nDelta Theta:")
    print(dtheta)
    print(dtheta.shape)
    print("\nTheta old:")
    print(theta)
    # Add the change in theta to old theta parameters
    thetaNew = np.add(theta,dtheta)
    print("\nTheta New:")
    print(thetaNew)
    # Calcualte new error with updated theta parameters
    e3 = calsim(y,thetaNew,na,nb)
    # Calculate new SSE term 
    SSEn = float(np.dot(e3.T,e3))
    print("\nSSE New:",SSEn)

    return SSEn,dtheta,thetaNew


# In[254]:


# 
N = 5000
mean = 0
std = 1
delta = 1e-6
mu = 0.1
muf = 10
muMax = 10e10
epsilon = 0.001
np.random.seed(42)
# create white noise/ error
e = std * np.random.randn(N) + mean


# In[265]:


# For ARMA(1,0)
num_true = [1,0]
den_true = [1,-.5]



# In[266]:


# Create the synthetic data to which we want to estimate parameters
system = (num_true,den_true,1)
x,y = signal.dlsim(system,e)


# In[267]:


# Step 0 intialize theta
theta_true,theta, na, nb = step0(num_true,den_true)
# Construct matrix X, A, and g
A,g,SSEo = step1(y,theta,delta,na,nb)
SSEn, dtheta, thetaNew = step2(y,theta,A,g,mu,na,nb)


# In[268]:


# step 3 Convergence

def step3(y,thetaNew,dtheta,SSEn,SSEo,A,g,mu,muf,muMax,delta,na,nb,epsilon):
    MAX_ITER = 4
    iterations = 0
    SSEs = [SSEo]

    while iterations < MAX_ITER:
        print("\nITERATION:",iterations)
        print(SSEn,"vs",SSEo)
        print("NORM2",np.linalg.norm(dtheta,2),"EPSILON",epsilon)
        if SSEn < SSEo:
            if np.linalg.norm(dtheta,2) < epsilon:

                theta = thetaNew
                variance_e = SSEn/ len(e) - (na + nb)
                cov_theta = np.multiply(variance_e,np.linalg.inv(A))

                print("\nEstimated Results")
                print("Estimated Theta:")
                print(theta)
                print("Variance of Error:", variance_e)
                print("Covariance of Estimated Theta", cov_theta)
                print("END")

                return theta, variance_e, cov_theta

            else:
                print("Not Converged")
                theta = thetaNew
                print("mu old:",mu)
                mu /= 10
                print("Mu new",mu)

        while SSEn >= SSEo:
            print("mu old:",mu)
            mu *= 10
            print("mu new:",mu)
            if mu > muMax:
                print("ERROR")

            return None
            SSEn, dtheta, thetaNew = step2(y,theta,A,g,mu,na,nb)
        iterations += 1

        if iterations > MAX_ITER:
            print("ERROR")

            return None
        
        print("Theta",thetaNew)
        print(mu)
        theta = thetaNew

        A,g,SSEo = step1(y,theta,delta,na,nb)

        SSEn, dtheta, thetaNew = step2(y,theta,A,g,mu,na,nb)

                                   
                                



# In[269]:


step3(y,thetaNew,dtheta,SSEn,SSEo,A,g,mu,muf,muMax,delta,na,nb,epsilon)


# In[ ]:




