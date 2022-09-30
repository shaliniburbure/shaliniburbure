#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


startup = pd.read_csv('50_startups.csv')
startup


# In[3]:


startup.shape


# In[4]:


startup.info()


# In[5]:


startup.describe()


# In[6]:


startup.isna().sum()


# In[7]:


startup01 = pd.get_dummies(startup,["State"])
startup01


# In[8]:


startup01.corr()


# In[9]:


startup01 =startup01.drop("State_New York",axis=1)
startup01


# In[10]:


startup01.corr()


# In[11]:


sns.pairplot(startup01)


# In[12]:


df = startup01.rename({"R&D Spend":"RD","Administration":"Admin","Marketing Spend":"Marketing","State_California":"SC","State_Florida":"SF"},axis=1)
df


# In[13]:


Y = df['Profit']
Y


# In[15]:


X = df.loc[ : , df.columns != 'Profit']
X


# In[ ]:


#Building of model


# In[17]:


model = smf.ols("Profit~RD+Admin+Marketing+SC+SF",data=df).fit()
model.params


# In[18]:


model.fittedvalues


# In[19]:


np.round(model.pvalues,4)


# In[20]:


np.round(model.tvalues,4)


# In[21]:


model.summary()


# In[ ]:


#linear model with variables whose P-value is greater than 0.05


# In[23]:


model_A= smf.ols("Profit~Admin",data=df).fit()
np.round(model_A.pvalues,4)


# In[24]:


model_A.rsquared


# In[26]:


sns.regplot(x='Admin',y='Profit',data=df)


# In[32]:


model_m = smf.ols("Profit~Marketing",data=df).fit()
np.round(model_m.pvalues,4)


# In[33]:


model_m.rsquared


# In[34]:


sns.regplot(x='Marketing',y='Profit',data=df)


# In[36]:


model_C = smf.ols("Profit~SC",data=df).fit()
np.round(model_C.pvalues,4)


# In[37]:


model_C.rsquared


# In[38]:


model_F = smf.ols("Profit~SF",data=df).fit()
np.round(model_F.pvalues,4)


# In[39]:


model_F.rsquared


# In[ ]:


#SF has the highest p-value


# In[40]:


data = df.drop('SF',axis=1)
data


# In[43]:


model = smf.ols('Profit~Admin+Marketing+RD+SC',data=df).fit()
model.summary()


# In[44]:


model.rsquared


# In[ ]:


# Model Validation ~ Calculating VIF


# In[45]:


rsq_r = smf.ols("RD~Marketing+Admin+SC",data=df).fit().rsquared
vif_r = 1/(1-rsq_r)
vif_r


# In[46]:


rsq_m = smf.ols("Marketing~RD+Admin+SC",data=df).fit().rsquared
vif_m = 1/(1-rsq_m)
vif_m


# In[47]:


rsq_a = smf.ols("Admin~Marketing+RD+SC",data=df).fit().rsquared
vif_a = 1/(1-rsq_a)
vif_a


# In[48]:


rsq_c = smf.ols("SC~Marketing+Admin+RD",data=df).fit().rsquared
vif_c = 1/(1-rsq_c)
vif_c


# In[49]:


D = pd.DataFrame({"Variables":["RD","Admin","Marketing","SC"],"VIF":[vif_r,vif_a,vif_m,vif_c]})
D


# In[50]:


model.resid


# In[51]:


qqplot = sm.qqplot(model.resid,line='q')
plt.title("QQ plot of residuals")


# In[52]:


sns.boxplot(model.resid)


# In[53]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[57]:


plt.scatter(get_standardized_values(model.fittedvalues),get_standardized_values(model.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized fitted values")
plt.ylabel("Standardized residual values")


# In[60]:


fig = plt.figure(figsize=(13,6))
fig = sm.graphics.plot_regress_exog(model,"RD",fig=fig)


# In[61]:


fig = plt.figure(figsize=(13,6))
fig = sm.graphics.plot_regress_exog(model,"Admin",fig=fig)


# In[62]:


fig = plt.figure(figsize=(13,6))
fig = sm.graphics.plot_regress_exog(model,"Marketing",fig=fig)


# In[63]:


fig = plt.figure(figsize=(13,6))
fig = sm.graphics.plot_regress_exog(model,"SC",fig=fig)


# In[ ]:


#Outliers Cook's Distance


# In[64]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[65]:


fig = plt.subplots(figsize=(15,6))
plt.stem(np.arange(len(data)),np.round(c,3))
plt.show()


# In[66]:


(np.argmax(c),np.max(c))


# In[67]:


data[data.index.isin([48,49])]


# In[68]:


data1 = data.drop(data.index[[48,49]],axis=0)
data1


# In[69]:


model2 = smf.ols("Profit~RD+Admin+Marketing+SC",data=data1).fit()


# In[70]:


model_influence2 = model2.get_influence()
(c_2, _) = model_influence2.cooks_distance
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c_2,3))
plt.show()


# In[71]:


(np.argmax(c_2),np.max(c_2))


# In[72]:


data2 = data1.drop(data1.index[[45,46]],axis=0).reset_index()
data2


# In[73]:


data2 = data2.drop(['index'],axis=1)
data2


# In[74]:


model3 = smf.ols("Profit~RD+Admin+Marketing+SC",data=data2).fit()
model_influence3 = model3.get_influence()
(c_3, _) = model_influence3.cooks_distance
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data2)),np.round(c_3,3))
plt.show()


# In[75]:


(np.argmax(c_3),np.max(c_3))


# In[76]:


data3 = data2.drop(data2.index[[19]],axis=0).reset_index()
data3


# In[77]:


data3 = data3.drop(['index'],axis=1)
data3


# In[83]:


model4 = smf.ols("Profit~RD+Admin+Marketing+SC",data=data3).fit()
model_influence4 = model4.get_influence()
(c_4, _) = model_influence4.cooks_distance
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data3)),np.round(c_4,3))
plt.show()


# In[84]:


(np.argmax(c_4),np.max(c_4))


# In[85]:


model_final = smf.ols("Profit~RD+Admin+Marketing+SC",data=data3).fit()
(model_final.rsquared,model_final.aic)


# In[86]:


influence_plot(model_final)
plt.show()


# In[87]:


model_final.resid


# In[88]:


model_final.fittedvalues


# In[89]:


data3["Predicted"]=model_final.fittedvalues
data3["Errors"]=model_final.resid
data3


# In[ ]:




