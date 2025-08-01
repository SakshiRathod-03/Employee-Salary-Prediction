#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[4]:


sal_data=pd.read_csv('Dataset09-Employee-salary-prediction (1).csv')
sal_data.head()


# In[5]:


sal_data.shape


# In[7]:


sal_data.columns


# In[8]:


sal_data.columns=['Age','Gende','Degree','Job_Title','Experience_years','Salary']


# In[9]:


sal_data.head()


# In[10]:


sal_data.dtypes


# In[11]:


sal_data.info()


# In[12]:


sal_data[sal_data.duplicated()]


# In[13]:


sal_data[sal_data.duplicated()].shape


# In[14]:


sal_data1=sal_data.drop_duplicates(keep='first')
sal_data1.shape


# In[15]:


sal_data1.isnull().sum()


# In[16]:


sal_data1.dropna(how='any',inplace=True)


# In[17]:


sal_data1.shape


# In[18]:


sal_data1.head()


# In[19]:


sal_data1.describe()


# In[20]:


corr=sal_data1[['Age','Experience_years','Salary']].corr()
corr


# In[21]:


sns.heatmap(corr,annot=True)


# In[22]:


sal_data1['Degree'].value_counts()


# In[23]:


sal_data1['Degree'].value_counts().plot(kind='bar')


# In[24]:


sal_data1['Job_Title'].value_counts()


# In[25]:


sal_data1['Job_Title'].unique()


# In[38]:


sal_data1['Gende'].value_counts().plot(kind='bar')



# In[39]:


sal_data1['Age'].plot(kind='hist')


# In[40]:


sal_data1.Age.plot(kind='box')


# In[41]:


sal_data1.Experience_years.plot(kind='box')


# In[42]:


sal_data1.Salary.plot(kind='box')


# In[43]:


sal_data1.Salary.plot(kind='hist')


# In[44]:


sal_data1.head()


# In[45]:


from sklearn.preprocessing import LabelEncoder
Label_Encoder=LabelEncoder()


# In[46]:


sal_data1['Gende_Encode']=Label_Encoder.fit_transform(sal_data1['Gende'])


# In[47]:


sal_data1['Degree_Encode']=Label_Encoder.fit_transform(sal_data1['Degree'])


# In[48]:


sal_data1['Job_Title_Encode']=Label_Encoder.fit_transform(sal_data1['Job_Title'])


# In[50]:


sal_data1.head()


# In[53]:


from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()


# In[58]:


sal_data1['Age_scaled'] = std_scaler.fit_transform(sal_data1[['Age']])
sal_data1['Experience_years_scaled'] = std_scaler.fit_transform(sal_data1[['Experience_years']])


# In[59]:


sal_data1.head()


# In[62]:


X = sal_data1[['Age_scaled','Gende_Encode','Degree_Encode','Job_Title_Encode','Experience_years_scaled']]
y = sal_data1['Salary']


# In[63]:


X.head()


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[77]:


x_train.shape,y_train.shape


# In[78]:


x_test.shape,y_test.shape


# In[82]:


from sklearn.linear_model import LinearRegression


# In[83]:


Linear_regeression_model=LinearRegression()


# In[85]:


Linear_regeression_model.fit(x_train,y_train)


# In[86]:


y_pred_lr=Linear_regeression_model.predict(x_test)
y_pred_lr


# In[105]:


df = pd.DataFrame({'y_Actual':y_test,'y_Predicted':y_pred_lr})
df['Error']=df['y_Actual']-df['y_Predicted']
df['ads_error']=abs(df['Error'])
df


# In[107]:


Mean_absolute_Error = df['ads_error'].mean()
Mean_absolute_Error


# In[109]:


from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[110]:


r2_score(y_test,y_pred_lr)


# In[111]:


print(f'Accuracy of the model={round(r2_score(y_test,y_pred_lr),4)*100}%')


# In[112]:


round(mean_absolute_error(y_test,y_pred_lr),2)


# In[113]:


print(f"Mean Absolute Error = {round(mean_squared_error(y_test,y_pred_lr),2)}")


# In[114]:


mse = round(mean_squared_error(y_test,y_pred_lr),2)
mse


# In[115]:


print('Root Mean Squared Error(RMSE)=',mse**(0.5))


# In[116]:


Linear_regeression_model.coef_


# In[121]:


Linear_regeression_model.intercept_


# In[122]:


sal_data1.head()


# In[133]:


Age1 = std_scaler.transform([[49]])
Age=5.86448677
Gende = 0
Degree = 2
Job_Title = 22
Experience_years1 = std_scaler.transform([[15]])  # Note: Double square brackets
Experience_years = 0.74415815



# In[134]:


std_scaler.transform([[15]])[0][0]


# In[135]:


Emp_Salary = Linear_regeression_model.predict([[Age,Gende,Degree,Job_Title,Experience_years]])
Emp_Salary


# In[136]:


print("Salary of that Employee with above Attributes=",Emp_Salary[0])


# In[ ]:




