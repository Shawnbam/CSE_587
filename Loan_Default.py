#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import sklearn
import pickle

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# ## Raw data collection and processing into pandas data frame

# In[2]:


df = pd.read_csv("Loan_Default.csv")


# ## Do profiling to get overall idea about the data distribution

# In[3]:


#prof = ProfileReport(df, title = 'Loan Dataset', explorative = True)

#prof.to_file('my_report.html')


# ## Remove ID and year columns since ID is just unique identifier, and year is 2019 for all

# In[4]:


df = df.drop('ID', axis=1)
df = df.drop('year', axis=1)


# ## Checking for missing values from columns

# In[5]:


df = df.dropna(axis = 1)


# ## Checking for rows with missing values in the columns

# In[6]:


df = df.dropna(axis = 0)


# ## Checking if dataset has duplicates rows and droping them
# 

# In[7]:


df = df.drop_duplicates()


# ## Inorder to have data consistency converting the 'Region' column to have lowercase values

# In[8]:


df['Region'] = df['Region'].str.lower()

# df.head()


# ## The dataset has incorrect spelling for the the column 'Security_Type' - correcting this
# 

# In[9]:


df['Region'] = df['Region'].replace('indriect','indirect')

# df.columns


# ## Scaling the 'property_value' in the dataset to optimise the results
# 

# In[10]:


df[['loan_amount']] = StandardScaler().fit_transform(df[['loan_amount']])


# df.head()


# ## Define numeric data types

# In[11]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


# ## Checking for outliers

# In[12]:


for i in df.select_dtypes(include=numerics).columns:
    mean = np.mean(df.select_dtypes(include=numerics)[i], axis=0)
    sd = np.std(df.select_dtypes(include=numerics)[i], axis=0)
    if i in ['ID']:
        continue
    lt = mean - 3 * sd
    rt = mean + 3 * sd
    col = i
    df = df.query(("{0} >= @lt & {0} <= @rt".format(col)))

print(df.shape)


# ## Correlation Heatmap

# In[13]:


sns.heatmap(df.corr(), annot=True)


# ## Changing total_units from string format as 1U, 2U ... to 1, 2

# In[14]:


df['total_units'] = df['total_units'].astype(str).str.replace('U', '').astype(int)
print(df.shape)


# ## Logic to print graphs in groups of size of mx

# In[15]:


grp = 0
mx = 4
done = []


# In[16]:


# plot every feature
def categorical_feature_plot_with_target(feature):
    plt.figure(figsize=(10, 3))
    ax = sns.catplot(x='Status', col=feature, kind='count', data=df, palette="cool_r");
    # ax.xaxis.set_label_position('top')
    for ax in ax.axes.ravel():
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
    ax.set_title(f"Distribution of feature {feature} wrt Output", y = 1.1, loc = 'left')



# ## Skip the below columns to plot because of high cardinality leading to higher cognitive overload

# In[17]:


skip = ['loan_amount', 'Credit_Score', 'Status']


# ## for every mx number of columns, plot the box

# In[18]:


for i in df.columns:
    if i in skip or i in done:
        continue
    done.append(i)
    grp += 1
    categorical_feature_plot_with_target(i)
    if grp == mx:
        grp = 0
        break


# In[19]:


for i in df.columns:
    if i in skip or i in done:
        continue
    done.append(i)
    grp += 1
    categorical_feature_plot_with_target(i)
    if grp == mx:
        grp = 0
        break
# format validation


# In[20]:


for i in df.columns:
    if i in skip or i in done:
        continue
    done.append(i)
    grp += 1
    categorical_feature_plot_with_target(i)
    if grp == mx:
        grp = 0
        break


# In[21]:


for i in df.columns:
    if i in skip or i in done:
        continue
    done.append(i)
    grp += 1
    categorical_feature_plot_with_target(i)
    if grp == mx:
        grp = 0
        break


# ## Bar plot against Status 0

# In[22]:


status0 = df[df['Status'] == 0]
for i in status0.columns:
    plt.bar(status0[i].unique(), status0[i].value_counts())
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.title('Bar plot of ' + i + ' with Status 0')
    plt.show()


# 
# 
# 
# 
# 
# 
# 
# ## Bar plot against Status 1

# In[23]:


status1 = df[df['Status'] == 1]
for i in status1.columns:
    plt.bar(status1[i].unique(), status1[i].value_counts())
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.title('Bar plot of ' + i + ' with Status 1')
    plt.show()


# ## Pair Plot

# In[24]:


sns.pairplot(df.drop(['Status'], axis=1))
plt.title('Pair plot')
plt.show()


# In[25]:


status1 = df[df['Status'] == 1]
s = pd.Series(status1['Credit_Score'], name = 'as')
ax = s.plot.kde()


# In[26]:


status1 = df[df['Status'] == 0]
s = pd.Series(status1['Credit_Score'], name = 'as')
ax = s.plot.kde()


# In[27]:


status1 = df[df['Status'] == 1]
s = pd.Series(status1['loan_amount'])
ax = s.plot.kde()


# In[28]:


status1 = df[df['Status'] == 0]
s = pd.Series(status1['loan_amount'])
ax = s.plot.kde()


# In[29]:


#df.to_csv('final_dataset.csv')

df.columns


# In[30]:


#Encoding the categorical variables


# In[31]:


#Seperating the categorical and numerical columns based on the datatype

datatype = pd.DataFrame(df.dtypes).reset_index()
#print(datatype)
categorical = []
numerical = []
for i, j in zip(datatype['index'], datatype[0]):
    if j == 'object':
        categorical.append(i)
        
    else:
        numerical.append(i)
        
        
print(categorical)

print(numerical)

# Binary variables
binary_variables = ['Security_Type', 'co-applicant_credit_type', 'Secured_by',
               'lump_sum_payment', 'interest_only', 'construction_type', 'business_or_commercial',
               'open_credit', 'Credit_Worthiness','Status']


# In[32]:


binary_variables


# In[33]:


df.columns


# In[34]:


df[binary_variables]


# In[35]:


categorical


# In[36]:


BinaryEncoder = LabelEncoder()

#for i in binary_variables:
    #df[i] = labelEncoder.fit_transform(df[i])

BinaryEncoder.fit


# In[37]:


#labelEncoder = LabelEncoder()

#for i in binary_variables:
   # df[i] = labelEncoder.fit_transform(df[i])


# df.columns
# print("after cols")
df_categorical = df[categorical]
#df_categorical.drop(columns=binary_variables,axis = 1, inplace=True)

df_categorical.columns

#Using one hot encoding

oneHotEncoder_categorical = OneHotEncoder()
df_onehot = oneHotEncoder_categorical.fit_transform(df_categorical)
df_enc_categorical = pd.DataFrame(df_onehot.toarray())


# Column names
print(oneHotEncoder_categorical.categories_)

oneHotEncoder_categorical.categories_

cat_columns = ['Female', 'Joint', 'Male', 'Sex Not Available',
               'type1', 'type2', 'type3',
               'ir', 'pr', 'sr',
               'CIB', 'CRIF', 'EQUI', 'EXP',
               'north', 'north-east', 'central', 'south']

#df_enc_categorical.columns = cat_columns
df.drop(columns=df_categorical.columns, inplace=True)
# Concat
df_merged = pd.concat([df, df_enc_categorical], axis=1, join='inner')

print("done")

df_merged

df_merged.columns


# In[38]:


#Splitting the data into training set and test set


df_merged.columns = df_merged.columns.astype(str)

training_set, testing_set = train_test_split(df_merged, test_size=0.4, random_state=42)

y_train = training_set['Status']
X_train = training_set.drop(columns=['Status'], axis = 1)
y_test = testing_set['Status']
X_test = testing_set.drop(columns=['Status'])





# In[39]:


df_merged.columns
column_names = oneHotEncoder_categorical.get_feature_names_out()
print(column_names)


# In[40]:


X_train.shape


# In[41]:


print(X_test)


# # Model 1 - Random Forest Classifier

# In[42]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


X_test.shape


# In[71]:


randomForest = RandomForestClassifier()


# Fitting the model

randomForest.fit(X_train, y_train)

with open('randomForest.pkl', 'wb') as f:
    pickle.dump(randomForest, f)
# Predictions

predict = randomForest.predict(X_test)

#Plotting the confusion matrix

confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)
print(accuracy_score(y_test, predict))


print("done")


# In[ ]:


randomForestConfusionMatrix = confusionMatrix


# In[46]:


# Plotting the confusion matrix as heatmap
sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[47]:


print(classification_report(y_test,predict))


# # Model 2 - Logistic Regression

# In[48]:


logmodel = LogisticRegression(max_iter=100000)
logmodel.fit(X_train,y_train)
with open('logmodel.pkl', 'wb') as f:
    pickle.dump(logmodel, f)
predict = logmodel.predict(X_test)
cols = X_test.columns.tolist()

# print("itssss", cols)


# In[72]:


confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)
print(accuracy_score(y_test, predict))


# In[ ]:


logisticRegressionConfusionMatrix = confusionMatrix


# In[50]:


sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[51]:


print(classification_report(y_test,predict))


# # Model 3 - Neural Network
# 

# In[52]:


clf = MLPClassifier(random_state=1, hidden_layer_sizes=[10,10], max_iter=3000)
clf.fit(X_train, y_train)

with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
predict=clf.predict(X_test)


# In[53]:


confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)


# In[73]:


print(accuracy_score(y_test, predict))

accuracy = accuracy_score(y_test, predict)*100
print('Accuracy of Neural Networks model is equal ' + str(round(accuracy, 2)) + ' %.')
print(classification_report(y_test,predict))


# In[ ]:


neuralNetworkConfusionMatrix = confusionMatrix


# In[55]:


sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# # Model 4 - Naive Bayes Classifier

# In[56]:


NB = GaussianNB()
NB.fit(X_train,y_train)
with open('NB.pkl', 'wb') as f:
    pickle.dump(NB, f)
predict=NB.predict(X_test)


# In[57]:


cnt = 0
selected_rows = []
for p in range(len(predict)):
    if predict[p] == 1 and X_test.iloc[p][0] >= 0:
        selected_rows.append(X_test.iloc[p])

df = pd.DataFrame(selected_rows)

# Write the dataframe to a CSV file
df.to_csv('selected_rows.csv', index=False)


# In[74]:


confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)
print(accuracy_score(y_test, predict))


# In[ ]:


naiveBayesConfusionMatrix = confusionMatrix


# In[59]:


sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[60]:


print(classification_report(y_test,predict))


# # Model 5 - Gradient Boosting Classifier

# In[61]:


gdboost = GradientBoostingClassifier()
gdboost.fit(X_train,y_train)
with open('gdboost.pkl', 'wb') as f:
    pickle.dump(gdboost, f)
predict=gdboost.predict(X_test)


# In[75]:


confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)
print(accuracy_score(y_test, predict))


# In[ ]:


gdBoostConfusionMatrix = confusionMatrix


# In[63]:


sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[64]:


print(classification_report(y_test,predict))


# # Model 6 - Decision Tree Classification

# In[65]:


tree=DecisionTreeClassifier()
tree.fit(X_train, y_train)
with open('tree.pkl', 'wb') as f:
    pickle.dump(tree, f)
predict = tree.predict(X_test)


# In[76]:


confusionMatrix = confusion_matrix(y_test, predict)
print(confusionMatrix)
print(accuracy_score(y_test, predict))


# In[ ]:


decisionTreeConfusionMatrix = confusionMatrix


# In[67]:


sns.heatmap(confusionMatrix, annot=True, cmap='Greens')


plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[68]:


print(classification_report(y_test,predict))


# In[69]:


print("DONE")


# In[ ]:





# In[70]:


def getLogisticClassification(X_NEW_TEST):
    predict = logmodel.predict(X_NEW_TEST)
    return predict

def getNeuralNetworkClassification(X_NEW_TEST):
    predict = randomForest.predict(X_NEW_TEST)
    return predict

def getRandomForestClassification(X_NEW_TEST):
    predict = clf.predict(X_NEW_TEST)
    return predict

def getNaiveBaseClassification(X_NEW_TEST):
    predict = NB.predict(X_NEW_TEST)
    return predict

def getGDBoostClassification(X_NEW_TEST):
    predict = gdboost.predict(X_NEW_TEST)
    return predict

def getDecisionTreeClassification(X_NEW_TEST):
    predict = tree.predict(X_NEW_TEST)
    return predict


# In[ ]:





# In[ ]:




