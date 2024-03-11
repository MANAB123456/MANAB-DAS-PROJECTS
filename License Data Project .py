#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[2]:


df=pd.read_csv("License_Data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# # Exploratory Data Analysis

# In[8]:


# Check the target variable distribution
sns.countplot(x='LICENSE STATUS', data=df)
plt.title('Target Variable Distribution')
plt.show()


# In[9]:


# Check the correlation between features
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()


# In[10]:


#Plot histograms of the numerical columns
df.hist(bins=20, figsize=(20,15))
plt.show()


# In[11]:


# Plot a correlation matrix of the numerical columns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()


# In[12]:


# Plot a box plot of the target variable and numerical columns
for col in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='LICENSE STATUS', y=col, data=df)
    plt.title(col)
    plt.show()


# # Handle missing values in the dataset

# In[13]:


# Check for missing values in the dataset
print('Missing values:', df.isnull().sum())


# In[14]:


# Drop unnecessary columns
df = df.drop(['ID', 'LICENSE DESCRIPTION', 'APPLICATION REQUIREMENTS COMPLETE', 'PAYMENT DATE', 'LICENSE STATUS CHANGE DATE'], axis=1)


# In[15]:


df.head()


# In[16]:


# Replace missing values in the 'LICENSE CODE' column with the mode
df['LICENSE CODE'].fillna(df['LICENSE CODE'].mode()[0], inplace=True)

# Replace missing values in the 'WARD' column with the median
df['WARD'].fillna(df['WARD'].median(), inplace=True)


# In[17]:


df.head()


# In[18]:


# Replace missing LATITUDE, LONGITUDE, and LOCATION values with the mean
df[['LATITUDE', 'LONGITUDE']] = df[['LATITUDE', 'LONGITUDE']].fillna(df[['LATITUDE', 'LONGITUDE']].mean())
df['LOCATION'].fillna(df['LOCATION'].mode()[0], inplace=True)


# In[19]:


df.head()


# In[20]:


df.isnull().sum()


# In[21]:


# Identify the numeric columns with missing values
numeric_cols_with_missing = [col for col in df.select_dtypes(include='number').columns if df[col].isna().any()]


# In[22]:


# Replace missing values with the mean
df[numeric_cols_with_missing] = df[numeric_cols_with_missing].fillna(df[numeric_cols_with_missing].mean())


# In[23]:


df.isnull().sum()


# In[24]:


# Identify the categorical columns with missing values
cat_cols_with_missing = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].isna().any()]


# In[25]:


# Replace missing values with the mode
df[cat_cols_with_missing] = df[cat_cols_with_missing].fillna(df[cat_cols_with_missing].mode().iloc[0])


# In[26]:


df.isnull().sum()


# # OUTLIER DETECTION

# In[27]:


numeric_cols = df.select_dtypes(include='number').columns


# In[28]:


# IDENTIFY OUTLIERS USING BOXPLOT

fig, ax = plt.subplots(len(numeric_cols), figsize=(10, 30))

for i, col in enumerate(numeric_cols):
    sns.boxplot(df[col], ax=ax[i])
    ax[i].set_title(col)


# In[29]:


# Identify outliers using IQR method:

outlier_indexes = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5*iqr
    lower_limit = q1 - 1.5*iqr
    outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)].index
    outlier_indexes.extend(outliers)

outlier_indexes = list(set(outlier_indexes))


# In[30]:


# Remove outliers:

df = df.drop(outlier_indexes)


# In[31]:


df.shape


# # DATA IMBALANCE

# In[32]:


from sklearn.utils import resample


# In[33]:


# Check class distribution:

class_counts = df['LICENSE STATUS'].value_counts()
print(class_counts)


# In[34]:


# Separate majority and minority classes
majority_class = df[df['LICENSE STATUS'] == 0]
minority_class = df[df['LICENSE STATUS'] == 1]

# Downsample majority class
majority_downsampled = resample(majority_class, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(minority_class), # match minority n
                                 random_state=42)  # reproducible results

# Combine minority class with downsampled majority class
balanced_df = pd.concat([majority_downsampled, minority_class])


# In[35]:


df.head()


# # Data Encoding

# In[36]:


from sklearn.preprocessing import LabelEncoder

# Select the categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# Create a LabelEncoder object for each categorical column and transform the data
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# In[37]:


df.head()


# In[38]:


df['CITY'].value_counts()


# # MODEL BUILDING

# In[39]:


# Split the dataset into train and test sets
X = df.drop('LICENSE STATUS', axis=1)
y = df['LICENSE STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest Classifier:')
print(classification_report(y_test, y_pred_rf))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))


# In[41]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred_rf)


# In[42]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred_rf)

print (accuracy)


# In[43]:


# AdaBoost
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print('AdaBoost Classifier:')
print(classification_report(y_test, y_pred_ada))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_ada))



# In[44]:


# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print('XGBoost Classifier:')
print(classification_report(y_test, y_pred_xgb))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_xgb))


# In[45]:


# Feature importance
# Print the feature importance of the random forest classifier
print('Random Forest Classifier Feature Importance:')
print(rf.feature_importances_)


# In[ ]:





# In[ ]:




