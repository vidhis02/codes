#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


airport = pd.read_csv("airports.csv")


# In[3]:


carrier = pd.read_csv("carriers.csv")
planes = pd.read_csv("plane-data.csv")


# In[4]:


df_2005 = pd.read_csv("20051.csv")
df_2006 = pd.read_csv("20061.csv")
df= pd.concat([df_2005,df_2006],ignore_index=True)# Combining 2005 and 2006 data 


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# In[8]:


pd.DataFrame({'unique values':df.nunique(),'missing values': df.isna().sum(),'different data types':df.dtypes})#Dataframe summary


# In[9]:


# Impute NA values using interpolation
df['DepTime'] = df['DepTime'].interpolate(method='linear', axis=0, limit_direction='both')
df['ArrTime'] = df['ArrTime'].interpolate(method='linear', axis=0, limit_direction='both')
df['ActualElapsedTime'] = df['ActualElapsedTime'].interpolate(method='linear', axis=0, limit_direction='both')
df['CRSElapsedTime'] = df['CRSElapsedTime'].interpolate(method='linear', axis=0, limit_direction='both')
df['AirTime'] = df['AirTime'].interpolate(method='linear', axis=0, limit_direction='both')
df['ArrDelay'] = df['ArrDelay'].interpolate(method='linear', axis=0, limit_direction='both')
df['DepDelay'] = df['DepDelay'].interpolate(method='linear', axis=0, limit_direction='both')


# In[10]:


#Renaming the "Month" to name 
import calendar
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
df.loc[df['Month'] == 1, ['Month']] = 'Jan'
df.loc[df['Month'] == 2, ['Month']] = 'Feb'
df.loc[df['Month'] == 3, ['Month']] = 'Mar'
df.loc[df['Month'] == 4, ['Month']] = 'Apr'
df.loc[df['Month'] == 5, ['Month']] = 'May'
df.loc[df['Month'] == 6, ['Month']] = 'Jun'
df.loc[df['Month'] == 7, ['Month']] = 'Jul'
df.loc[df['Month'] == 8, ['Month']] = 'Aug'
df.loc[df['Month'] == 9, ['Month']] = 'Sep'
df.loc[df['Month'] == 10, ['Month']] = 'Oct'
df.loc[df['Month'] == 11, ['Month']] = 'Nov'
df.loc[df['Month'] == 12, ['Month']] = 'Dec'


# In[11]:


# Renaming 'DayOfWeek' to their abbreviated name manually 
df.loc[df['DayOfWeek'] == 1, ['DayOfWeek']] = 'Mon'
df.loc[df['DayOfWeek'] == 2, ['DayOfWeek']] = 'Tue'
df.loc[df['DayOfWeek'] == 3, ['DayOfWeek']] = 'Wed'
df.loc[df['DayOfWeek'] == 4, ['DayOfWeek']] = 'Thu'
df.loc[df['DayOfWeek'] == 5, ['DayOfWeek']] = 'Fri'
df.loc[df['DayOfWeek'] == 6, ['DayOfWeek']] = 'Sat'
df.loc[df['DayOfWeek'] == 7, ['DayOfWeek']] = 'Sun'


# In[12]:


df.head()


# In[13]:


#Q1.When is the best time of day, day of the week, and time of year to fly to minimise delays?
pd.set_option("display.max_columns",30)
pd.set_option("display.max_rows",30)
df1 = df[['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','DepDelay','ArrDelay']]
df1.head()


# In[14]:


# Define a function to convert intergers/float into time format
def Timeformat(i):
    try:
        if i != 0 or i != np.NAN:
            if str(int(i))[:-2].zfill(2) == '24':
                hh = '00'
            else:
                hh = str(int(i))[:-2].zfill(2)
            
            mm = str(int(i))[-2:].zfill(2)
            time = f"{hh}:{mm}"
        return time
    except Exception as e:
        pass


# In[15]:


# Convering data into time format
df1['CRSDepTime'] = df1['CRSDepTime'].apply(Timeformat)
df1['CRSArrTime'] = df1['CRSArrTime'].apply(Timeformat)
df1[['CRSDepTime', 'CRSArrTime']].head()


# In[16]:


# Slicing the Departure time to hour format
df1['CRSDepHour'] = df1['CRSDepTime'].apply(lambda x: x[:2])
df1['CRSArrHour'] = df1['CRSArrTime'].apply(lambda x: x[:2])


# In[17]:


df1[['CRSDepTime', 'CRSArrTime', 'CRSDepHour']].head() #printing dataset in order to validate results


# In[18]:


# Distribution of DepDelay & ArrDelay vs Scheduled Departure Time
ax = df1.groupby('CRSDepHour')[['DepDelay','ArrDelay']].mean().plot.bar(rot=0, alpha=0.8, figsize=[18,9])
ax.set_ylabel("Delay in Arrival and Departure time",fontsize=11)
ax.set_xlabel("Scheduled Departure Time",fontsize=12)
ax.set_xticklabels(['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00',
                    '12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'])
ax.legend(['Departure Delay','Arrival Delay'], fontsize = 12)
ax.set_title("Departure & Arrival Delay vs Scheduled Departure Time")


# In[19]:


#Departure and Arrival Delay vs Days of Week
daysofweek = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
ax = df1.groupby('DayOfWeek')[['DepDelay','ArrDelay']].mean().reindex(daysofweek).plot(kind='bar',alpha=0.8, figsize=[14,6])
ax.set_ylabel("Departure & Arrival Delays in Minutes)",fontsize=12)
ax.set_xlabel("Day of Week",fontsize=12)
ax.legend(['Departure Delay','Arrival Delay'], fontsize=10 )
ax.set_title("Departure & Arrival Delay vs Days of the Week")


# In[20]:


#Distribution of Departure & Arrival Delay Vs Month
daysofmonth = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax = df1.groupby('Month')[['DepDelay','ArrDelay']].mean().reindex(daysofmonth).plot(kind='bar',alpha=0.8, figsize=[14,6])
ax.set_ylabel("Departure & Arrival Delays (Minutes)",fontsize=12)
ax.set_xlabel("Month",fontsize=12)
ax.legend(['Departure Delay','Arrival Delay'], fontsize = 10)
ax.set_title("Distribution of Departure & Arrival Delay vs Month")


# In[21]:


#Q2.Do older planes suffer more delays?
planes.head(40) #Since most of the data is NaN in the first few rows


# In[22]:


# Check for NA Values
planes.isnull().sum()


# In[23]:


# Remove NA Values
df_planes = planes.dropna(axis=0)
df_planes.head()


# In[24]:


# Merging of df with df_planes dataset
df_planes = planes.rename(columns = {'year': 'YearOfManufacture','tailnum': 'TailNum'})
df2_planes = df.merge(df_planes[['TailNum','YearOfManufacture']],how='left',left_on='TailNum',right_on='TailNum')  
df2_planes.head()


# In[25]:


# Sum of Depature Delay and Arrival Delay respectively
df2planes_sum = df2_planes.groupby('YearOfManufacture')[['DepDelay','ArrDelay']].sum()


# In[26]:


df2planes_sum.head()
df2planes_sum = df2planes_sum.reset_index()


# In[27]:


g = sns.catplot(x="YearOfManufacture", y="ArrDelay", kind="bar", data=df2planes_sum, height=60, aspect=8.27, margin_titles=True)
g.fig.suptitle('Distribution of Arrival Delays vs Year of Manufacture')


# In[28]:


sns.catplot(x="YearOfManufacture", y="DepDelay", kind="bar", data=df2planes_sum, height=60, aspect=8.27)
g.fig.suptitle('Distribution of Departure Delays vs Year of Manufacture')
ax.set_ylabel("Departure Delays",fontsize=12)
ax.set_xlabel("year",fontsize=12)


# In[29]:


#Q3:How does the number of people flying between different locations change over time?
# Subset datetime variables, Origin and Destination locations
df3 = df[['Year','Month','DayofMonth','DayOfWeek','Origin','Dest']]
df3.head()


# In[30]:


airport.head()


# In[31]:


# Finding the count of people travelling from the city of Boston(BOS) to Chicago(ORD)
df3_location =df3[(df3['Origin'] == 'BOS') &
              (df3['Dest'] == 'ORD')]
df3_location.count()


# In[32]:


#Plotting out all the graphs
fig, axs = plt.subplots(2, 2, figsize=(18, 10))
axs[0, 0].plot(df3_location.groupby(['Year']).count(), color='mediumblue')
axs[0, 0].set_xlabel('Year', fontsize=14, weight = "bold")
axs[0, 1].plot(df3_location.groupby(['Month']).count().reindex(daysofmonth), color='mediumblue')
axs[0, 1].set_xlabel('Month', fontsize=14, weight = "bold")
axs[1, 0].plot(df3_location.groupby(['DayofMonth']).count(), color='mediumblue')
axs[1, 0].set_xlabel('DayofMonth', fontsize=14, weight = "bold")
axs[1, 1].plot(df3_location.groupby(['DayOfWeek']).count().reindex(daysofweek), color='mediumblue')
axs[1, 1].set_xlabel('DayOfWeek', fontsize=14, weight = "bold")
plt.suptitle('Number of people travelling from BOS to ORD', fontsize=20, weight = "bold")
plt.tight_layout()
plt.show()


# In[51]:


#Q4. Can you detect cascading failures as delays in one airport create delays in others?
# Make a copy of dataset
df4 = df.copy(deep=True) 


# In[52]:


from time import strptime
df4['Month'] = df4['Month'].apply(lambda x: strptime(x,'%b').tm_mon)


# In[53]:


# Adding a date column to the dataset
df4['Date'] = pd.to_datetime(dict(year=df4.Year, month=df4.Month, day=df4.DayofMonth))
first_col = df4.pop('Date')
df4.insert(0,'Date', first_col)


# In[54]:


df4.head()


# In[62]:


# Subsetting columns required to test for cascading failures
df4_data = df4[['Date','UniqueCarrier','TailNum','Origin','Dest','Distance','CRSDepTime','DepTime','DepDelay','CRSArrTime','ArrTime','ArrDelay']]
df4_data.head()


# In[63]:


df4_analysis = df4_data[(df4_data['ArrDelay'] > 0) & (df4_data['DepDelay'] > 0)]


# In[67]:


df4_analysis[(df4_analysis['TailNum']=='N326UA')].head(2)


# In[68]:


df4_analysis[(df4_analysis['Origin'] == "ORD") & (df4_analysis['TailNum']=='N326UA')].head(2)


# In[59]:


#Q5. Use the available variables to construct a model that predicts delays
# Import machine learning model for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


# In[42]:


df5 = df[['Year','DayofMonth','DepTime','CRSDepTime','DepDelay','Distance','TaxiIn','TaxiOut','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay']]
df5.head()


# In[43]:


# Subset the main dataframe 
df5_model = df5.head(10000)


# In[44]:


df5_model['DepartureDelay'] = 0
df5_model['DepartureDelay'].mask(df5_model['DepDelay'] > 0, 1, inplace=True)


# In[45]:


# Seperating our variables into dependent and independent variables
X = df5_model.drop(["DepartureDelay"],axis =1) # Independent Variables
y = df5_model["DepartureDelay"].values         # Dependent Variable


# In[46]:


# Splitting our data into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[47]:


# Model development using the Logistic Regression Module
LR = LogisticRegression()
LR.fit(X_train,y_train) # Fitting the model with the data
y_predict = LR.predict(X_test) # Performing prediction on the testset


# In[48]:


# Model Evaluation using Confusion Matrix
conf = metrics.confusion_matrix(y_test, y_predict)
conf


# In[49]:


# Checking Classification Report
model_summary = classification_report(y_test, y_predict)
print(model_summary)


# In[50]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
print("Precision:",metrics.precision_score(y_test, y_predict))
print("Recall:",metrics.recall_score(y_test, y_predict))


# In[ ]:




