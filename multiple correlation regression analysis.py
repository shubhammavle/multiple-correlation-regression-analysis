# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:53:06 2024

@author: Shubham Sharad Mavle
"""

import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv('cars.csv')
#eda
#1.measure the central tendancy

cars.describe()
#graphical representation
import matplotlib.pyplot as plt
plt.bar( height=cars.HP,x=np.arange(1,82,1))
sns.displot(cars.HP)
#data  is right skewed
plt.boxplot(cars.HP)
#there are sevral outliers in HP col
#similar op are expected for other 3 col
sns.displot(cars.MPG)

plt.boxplot(cars.MPG)

sns.displot(cars.VOL)
plt.boxplot(cars.VOL)
sns.displot(cars.SP)
plt.boxplot(cars.SP)
sns.displot(cars.WT)
plt.boxplot(cars.WT)
#now lets plot joint plot joint  plot is to show scatter plot
#histograme
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])
#now ler us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'],)
#

##QQ plot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist='norm',plot=pylab)
plt.show()
#MPG data is normally distributed
#there are 10 scarrer plot need to be plotted on by one to 
#plot so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#linearity: direction : and strength:
#you can check the collinearity problem between the input
#you can check plot between sp and hp they are strongly conne

#now let us check r value between variables
cars.corr()
#you can check sp and hp ,r vlaue is 0.97 ans same way
#you can check wt and vol it has got 0999

#linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary
#R square value observed is 0.771<0.85
#p-values of WT and vol is 0.814 and 0.556 which is very high
#it means it is greter than 0.05 wt and vol columns
#we need to ignore or delete .insted deletion 81 entries
#let us check row wise outliers 
#identifying is there any infuential value to check you can
#use influential index

import statsmodels.api as sm
sm.graphics.influence_plot
#76 is the value which has got outliers
#go to data frame and check 76 th entry
#let us delete the entry
car_new=cars.drop(cars.index[[76]])

#again apply the regression to cars_new
ml_new=smf.ols('MPG~WT+VOL+SP+HP',data=cars_new).fit()
ml_new.summary()
#r square is 0.819 but p values are same hence not scatter plot
#now next option is delete the col but que is which col is to be deleted
#we alredy check correlation factor r
#WT is less hence can be deleted

#another approach is to check the collinearity,r square is giving that value
#we will have to apoply regression w.r.t.xl and input
#as x2,x3 and x4 so on so forth

rsq_hp=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
#vif is varience influential factor 

rsq_wt=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_hp)

rsq_vol=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_hp)

rsq_sp=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit().rsquared
vif_sp=1/(1-rsq_hp)

### storing the value in data frame
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

### let us drop WT and apply correlation to remaing three
final_ml=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
final_ml.summary()
#R square is 0.770 and p values 0.00,0.012 <0.05
#prediction
pred=final_ml.predict(cars)

##QQ plot 
res=final_ml.resid
sm.qqplot(res)
plt.show()
stats.probplot(res,dist='norm',plot=pylab)
plt.show()

#let us plot the residual plot which takes the residuals variable and data
sns.residplot(x=pred,y=cars.MPG,lowess=(True))
plt.xlable('Fitted')
plt.ylable('Residual')
plt.title('Fitted vs Residual')











