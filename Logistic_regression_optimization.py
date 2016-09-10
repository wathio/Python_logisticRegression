# -*- coding: utf-8 -*-
"""
Linear regression and cost function optimization
"""

import numpy as np
import matplotlib.pyplot as plt
#set the path below to the correct directory.
dir="set The path to the correct directory "




x=np.loadtxt(dir+"ex2x.dat")
y=np.loadtxt(dir+"ex2y.dat")
x0=np.ones(len(x))
#defining some variables
theta_0,theta_1=0,0  #initializing parameters for theta
m=len(y)             #number of training set
alpha=0.07          # learning rate
num_iter=1500       #max number of terations
theta_opt0=[]       #list to store theta_0 after each iterations
theta_opt1=[]       #list to store theta_1 -||-----||---------
J=[]                #list to store j(theta) --||-------||-----
# parameters to optimize
theta=np.array([theta_0,theta_1])



#Defining the learning function
def h(x):
    
    return (theta_0+theta_1*x)


#defining the cost function J(theta)

def j(theta,x,y,m):
    
    sum_sqrt=0    
    for i in range(m):
       # print(h(x[i]))
        sum_sqrt+=0.5*((h(x[i]) - y[i])**2)/(m)
    return(sum_sqrt)
    
j0=j(theta,x,y,50)

print('*'*10)
print("j0= %f"%j0)
print('*'*10)

###################################################
## Implementing the gradient descent algorithm
###################################################

## Computing the gradient of the cost function
## parameter a helps to select the column of ones in the design matrix()
def delta_j(alpha,x,y,m,a):
    sum_delta=0
    for i in range(m):
        if a==0:
            sum_delta += alpha*(h(x[i]) - y[i])*x0[i]/m
        else:
           sum_delta += alpha*(h(x[i]) - y[i])*x[i]/m 
    return(sum_delta)

t0=delta_j(alpha,x,y,m,1)
print('@'*10)
print("t0= %f"%t0)
print('@'*10)

##Optimizing the cost function j(theta) by updating
## theta_0 and theta_1 for num_iter  iterations

for iteration in range(num_iter):
    temp0= theta_0-delta_j(alpha,x,y,m,0)
    temp1= theta_1-delta_j(alpha,x,y,m,1)
    theta_0=temp0
    theta_1=temp1
    theta=np.array([theta_0,theta_1])
    theta_opt0.append(theta_0)
    theta_opt1.append(theta_1)
    J.append(j(theta,x,y,m))
#print(t   
print("theta0= %f , theta1= %f , j(theta)= %f "%(theta_0,theta_1,j(theta,x,y,m)))

#############################################################
## Plotting training data and the learning function 
## on the same figure
#############################################################
plt.plot(x,y,'ro',x,h(x),'g-')
plt.xlabel("Ages(years)")
plt.ylabel("Heigths(m)")
    
##########################################
## Vectorization of the cost function
##########################################
#Uncomment the lines below to use the vectorized form
#######################################################
#x=np.matrix(np.loadtxt(dir+"ex2x.dat"))
#y=np.matrix(np.loadtxt(dir+"ex2y.dat"))
#theta_0,theta_1=0,0
#m=np.size(y)
#x0=np.ones((m,1))
#adding one column of ones in the X matrix for multiplication with theta_0
#X=np.c_[x0,x.getT()]
#j=0.5*(X*theta.getT()-y.getT()).getT()*(X*theta.getT()-y.getT())/m
#print(j) 
