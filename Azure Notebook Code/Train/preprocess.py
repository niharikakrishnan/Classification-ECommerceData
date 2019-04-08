
# coding: utf-8

# # Customer Classification using E-Commerce Dataset
# 

# ## About Dataset
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

# ## Aim
# We aim to implement various classification algorithms like SVM, Logistic Regression, Naive Bayes, Random Forest, SGD, k-NN to predict a customer's origin and to compare the performance of these supervised machine learning models.

# ### 1. Data Processing

# Importing necessary libraries for data preprocessing, data augmentation and classification

# In[4]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import nltk
warnings.filterwarnings('ignore')


# Load comma separated value dataset. File should be present in the same directory else path must be sent as an argument
df = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

# Function "information" to display the description, head and dtypes of the dataset. Function "shape" that returns the shape of the dataset. Functions can be called at any instance of the dataset
def information(df):
    print(df.describe())
    print(df.dtypes)
    print(df.head())

def shape(df):
    print(df.shape)

# Creating new columns by extracting Year, Month, Day and Hour from TimeStamp ~ Inference: Helps to bucketize and perform analysis on the basis of month, day and hour of the purchase
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month 
df['Day'] = df['InvoiceDate'].dt.day 
df['Hour'] = df['InvoiceDate'].dt.hour 

# Drop column "Invoice Date" from dataset since the individual features are extracted. Shape of the dataset: (541909, 11)
df.drop(columns=['InvoiceDate'], inplace=True)

# Drop duplicates by keeping the first value. Inference - 5269 rows are removed from the dataset
df.drop_duplicates(keep='first', inplace=True)

# Another column in the data is added which gives the total amount spent per transaction. Usecase: Analysis of amount spent by each country, customer
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']

# Check and display the total number of rows that have missing values in each column. Usecase: Data needs to synthesised or removed based on the below figures
df.isnull().sum().sort_values(ascending=False)

# Count the total number unique values in each column of the dataset. 
# Inference: No of countries is 38. Hence, it is a supervised multi-class classification problem
df.nunique()

# 2. Exploratory Data Analysis
# #### Exploring the content of variables
# 
# This dataframe contains 8 variables that correspond to:
# <br><br>
# __InvoiceNo:__ Invoice number. Nominal - A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. <br>
# __StockCode:__ Product code. Nominal - A 5-digit integral number uniquely assigned to each distinct product.<br>
# __Description:__ Product (item) name. Nominal. 
# <br>
# __Quantity:__ Numeric - The quantities of each product (item) per transaction. <br>
# __InvoiceDate:__ Invice Date and time. Numeric - The day and time when each transaction was generated. <br>
# __UnitPrice:__ Unit price. Numeric - Price per unit of the product <br>
# __CustomerID:__ Customer number. Nominal - A 5-digit integral number uniquely assigned to each customer. <br>
# __Country:__ Country name. Nominal - The name of the country where each customer resides.<br>

# #### Exploring each feature of the dataset
# 
# There are totally 12 features: Quantity, Unit Price, CustomerID, Year, Month, Day, Hour, Total Price, Country, Description, Stock Code, Invoice No 

def correlaton(df):
  # ##### 1. Correlation of Numeric features in the dataset

  # In[182]:

  df.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


def eda_country(df):
  # ##### 2. Exploratory Data Analysis with respect to each country

  # Number of orders made by each country. Inference - People from United Kingdom purchase the most 

  # In[183]:


  total_orders=df.groupby('Country')['Quantity'].count().sort_values(ascending=False)
  total_orders.plot('bar')
  plt.xlabel('Country')
  plt.ylabel('Number of Orders')
  plt.title('Number of Orders per Country', fontsize=16)
  plt.show()


  # No.of Invoices per country. Inference - People from United Kingdom have the most visits/transactions

  # In[184]:


  No_invoice_per_country = df.groupby(["Country"])["InvoiceNo"].count().sort_values()
  No_invoice_per_country.plot(kind='barh', figsize=(15,12))
  plt.title("Number of Invoices per Country")
  plt.show()


  # Best Buyer with respect to country. 

  # In[185]:


  best_buyer = df.groupby(["Country", "InvoiceNo"])["TotalPrice"].sum().reset_index().groupby(["Country"])["TotalPrice"].mean().sort_values()
  best_buyer.plot(kind='barh', figsize=(15,12))
  plt.title("Average Basket Price per Country")
  plt.show()


def eda_quantity(df):
  # ##### 3. Exploratory Data Analysis with respect to Quantity

  # Using violin plot to get the distribution of quantity data its probability density. 
  # 
  # Inference: Value of quantity is skewed - Max amd min have values around 80000 which has adverse effects with the training data

  # In[186]:


  sns.set(style="whitegrid")
  sns.violinplot(x=df.Quantity)


  # Visualising total quantity. Shows outliers where quantity is less than -2000. 

  # In[187]:


  plt.figure(figsize=(15,15))
  x=df.Quantity.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(411) #1st digit #rows, 2nd digit #columns, 3rd digit plot number
  plt.scatter(x[0], x[1], marker='o')
  plt.title('Quantity plots',fontsize=15)
  plt.ylabel('Occurrence',fontsize=12)


  # Visualising quantity which are negative. -80000 to -70000 are clear outliers

  # In[188]:


  plt.figure(figsize=(15,15))
  x=df[df['Quantity']<0].Quantity.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(412)
  plt.scatter(x[0], x[1], marker='o')
  plt.ylabel('Occurrence')


  # Visualising quantity above 50000

  # In[189]:


  plt.figure(figsize=(15,15))
  x=df[df['Quantity']>10000].Quantity.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(413)
  plt.scatter(x[0], x[1], marker='o')
  plt.ylabel('Occurrence',fontsize=12)


# Identifying if the outliers have an equivalent counterpart ie same quantity value that has been bought and cancelled
# Inference - Two items satisfy the condition and are removed from the dataset

# In[190]:


# df[df['Quantity'].abs()>50000]
# Removing the outliers, that is the items which have more than 50000 or less than 50000 as quantity.
df.drop([61619, 61624, 540421, 540422], inplace=True)


# Identifying if there are similar conditions where quantity is more that 5000 and less than 50000. To also check if these are fake purchases with null unit price. 
# df[(df['Quantity'].abs()>5000) & (df['Quantity'].abs()<50000)]

# Removing the outliers
df.drop([4287, 74614, 115818, 225528, 225529, 225530, 502122], inplace=True)

def eda_unitprice(df):

  # ##### 4. Exploratory Data Analysis with respect to Unit Price (to find outliers)

  # Plotting the feature on of uit price to check for outliers and find out if they are genuine transactions. 
  # 
  # Inference - Unit price is as low as 10,000 and as high as 40,000. Let's find out

  # In[195]:
  sns.violinplot(df.UnitPrice)


  # Count the unit price and plot to check outliers. Inference - Show items which have less than 0 unit price which is not possible. let's check

  # In[196]:


  plt.figure(figsize=(15,15))
  x=df.UnitPrice.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(411)
  plt.scatter(x[0], x[1], marker='o')
  plt.title('UnitPrice plots',fontsize=15)
  plt.ylabel('Occurrence',fontsize=12)
  plt.show()


  # This shows that there are two rows which are non numeric since even after absolute, they are showing negative. 

  # In[197]:


  plt.figure(figsize=(15,15))
  x=df[df['UnitPrice'].abs()>10000].UnitPrice.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(412)
  plt.scatter(x[0], x[1], marker='o')
  plt.ylabel('Occurrence',fontsize=12)
  plt.show()


  # In[198]:


  df[df['UnitPrice']<0]


# Drop outliers since they negatively affect the model
df.drop([299983, 299984], inplace=True)


def eda_uniteprice1(df):
  # To find outliers where price is above 15000.

  # In[200]:


  plt.figure(figsize=(15,15))
  x=df[df['UnitPrice'].abs()>15000].UnitPrice.value_counts().reset_index().as_matrix().transpose()
  plt.subplot(413)
  plt.scatter(x[0], x[1], marker='o')
  plt.ylabel('Occurrence',fontsize=12)
  plt.show()


  # To find purchases where the unit price is aove 15000 pounds. General intuition that it is an outlier

  # In[201]:


  df[df['UnitPrice']>15000]


# In[202]:
#Hence removed the outliers
df.drop([222681, 524602, 43702, 43703], inplace=True)

def eda_unitprice2(df):
  # Just to check what items are above a unit price of 3000. Inference - Amazon Fee
  df[(df['UnitPrice']>3000)]
  # We count the negative value of of quantity and Unit Price
  # In[204]:
  print("The number of rows with negative Quantity:",sum(n < 0 for n in df.Quantity))
  print("The number of rows with negative UnitPrice:",sum(n < 0 for n in df.UnitPrice))


  # #### 5. Exploratory Data Analysis with respect to Customers and products

  # In[205]:


  pd.DataFrame([{'products': len(df['StockCode'].value_counts()),    
                 'transactions': len(df['InvoiceNo'].value_counts()),
                 'customers': len(df['CustomerID'].value_counts()),  
                }], columns = ['products', 'transactions', 'customers'], index = ['quantity'])


  # It can be seen that the data concern 4370 users and that they bought 4068 different products. The total number of transactions carried out is more than 21000

  # In[206]:


  #Number of products purchased per customer
  temp = df.groupby(by=['CustomerID'], as_index=False)['Quantity'].count()
  nb_products_per_basket = temp.rename(columns = {'InvoiceNo':'Quantity'})
  nb_products_per_basket.sort_values('Quantity', ascending=False)[:20]


  # Top selling products

  # In[207]:


  grps = np.array([['Month', 'Week'], ['Hour', 'Minute']])
  ctry = np.array([['United Kingdom', 'Japan'], ['Germany', 'France']])
  fltr = ['DOT', 'POST', 'M']
  top_n = 10

  fig, ax = plt.subplots(grps.shape[0],grps.shape[1], figsize=(14, 14))

  for i in range(0, ctry.shape[0]):
      for j in range(0, ctry.shape[1]):
          grp_data = df[df['Country'] == ctry[i,j]]
          grp_data = grp_data[~grp_data['StockCode'].isin(fltr)]
          grp_data = grp_data[['StockCode', 'TotalPrice']].groupby(['StockCode']).sum().sort_values(by='TotalPrice', ascending=False)        
          grp_data = grp_data[0:top_n]    
          grp_data = grp_data.reset_index()
          
          ax[i,j].barh(y=grp_data.index, width='TotalPrice', data=grp_data)
          ax[i,j].invert_yaxis()
          ax[i,j].set_yticks(range(0,top_n))
          ax[i,j].set_yticklabels(grp_data['StockCode'].tolist())
          ax[i,j].set_ylabel('Stock code')        
          ax[i,j].set_xlabel('TotalPrice')                
          ax[i,j].set_title('Top 10 ' + ctry[i,j])        
          
  plt.show()


  # #### 6. Exploratory Data Analysis for Stock Code

  # Stock code is predominantly numeric. However, some have characters which represent different type of transactions like Discount, Manual, Cancellation, etc. Use Regular expression to check what are the unique transactions of the stock code which contains characters

  # In[208]:


  list_special_codes = df[df['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
  list_special_codes


  # To check if the stock code is actually unique or not. inference - Stock code is not a unique value.

  # In[209]:


  df_stk=df.groupby("StockCode")['Description'].nunique().reset_index()
  df_stockcode=df_stk[df_stk["Description"]>3]
  df_stockcode


  # #### 7. Exploratory Data Analysis of Time Stamp

  # To find the number of purchases each month

  # In[210]:


  df.pivot_table(index=df['Month'],values='InvoiceNo',aggfunc=np.size).plot(kind='bar', figsize=(14,7),
                                                                            title='Frequency of orders(Monthly)');


  # To find the number of purchases each day  grouped by all months

  # In[211]:


  df.pivot_table(index=df['Day'],values='InvoiceNo',aggfunc=np.size).plot(kind='bar', figsize=(14,7),
                                                                          title='Frequency of orders(Monthly)');


  # To find the time at which maximum orders take place. Inference - No orders after 8pm and 6am

  # In[212]:


  df.pivot_table(index=df['Hour'],values='InvoiceNo',aggfunc=np.size).plot(kind='bar', figsize=(14,7),
                                                                           title='Frequency of orders(Monthly)');


  # Heat map graphical representation of data where the individual values ie Hour and Day are plotted to find which is the optimal period for shoppers
  # 
  # Inference - 21st day of the month at 3pm is the best. Major transactions happen between 12pm and 3pm

  # In[213]:


  week_vs_hour = df.pivot_table(index=df['Day'],values='InvoiceNo',columns=df['Hour'],aggfunc=np.size)
  plt.figure(figsize=(12,6))
  sns.heatmap(week_vs_hour,cmap='coolwarm',linecolor='white',linewidths=0.01);


# #### 8. Exploratory Data Analysis for Description

# Handle incorrect Description and remove them from the dataset
df = df[df["Description"].str.startswith("?") == False]
df = df[df["Description"].str.isupper() == True]
df = df[df["Description"].str.contains("LOST") == False]

def eda_description(df):
  # Find the maximum purchased products. Inference - White Hanging Herat T-Light Holder

  # In[215]:


  high_descrip = df.groupby('Description')['Quantity'].count().sort_values(ascending=False)[:10]
  high_descrip.plot('bar')
  plt.xlabel('Description')
  plt.ylabel('Total Quantity')
  plt.title('Items most purchased', fontsize=16)
  plt.show()


def eda_nullcustomer():
  # #### 8. Exploratory Data Analysis on Null Customer and imputing data

  # Analysing the percentage of not null customers in the dataset. Inference - 25.2% Null, 74.8% Not Null

  # In[216]:


  CustomerID_notnull = df.loc[~df['CustomerID'].isnull()] 
  CustomerID_null = df.loc[df['CustomerID'].isnull()] 
  pie_data = []
  pie_data.append(len(CustomerID_null))
  pie_data.append(len(CustomerID_notnull))
  plt.pie(pie_data, labels=['Null', 'Not Null'], autopct='%1.1f%%',)
  plt.show()


  # To find from which country the maximum number of Null occurs

  # In[217]:


  CustomerID_null.groupby(['Country']).size().sort_values(ascending=False)


  # To find the ratio of quantity of units purchased by null & not null customers

  # In[218]:


  month_null = (CustomerID_null.groupby(['Month'])['Quantity'].sum())
  month_notnull  = (CustomerID_notnull.groupby(['Month'])['Quantity'].sum())
  month = pd.DataFrame([month_null, month_notnull]).transpose()

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))
  month.plot.bar(stacked=True, ax=axes[0])
  month.plot.box(ax=axes[1])


  # In[219]:


  pie_data = []
  pie_data.append(CustomerID_null['Quantity'].sum())
  pie_data.append(CustomerID_notnull['Quantity'].sum())
  plt.pie(pie_data, labels=['Null', 'Non Null'], autopct='%1.1f%%',)
  plt.show()


  # To find the ratio of amount spent by null & not null customers

  # In[220]:


  CustomerID_notnull['Total_Amount'] = CustomerID_notnull['Quantity']*CustomerID_notnull['UnitPrice']
  CustomerID_null['Total_Amount'] = CustomerID_null['Quantity']*CustomerID_null['UnitPrice']
  pie_data = []
  pie_data.append(CustomerID_null['Total_Amount'].sum())
  pie_data.append(CustomerID_notnull['Total_Amount'].sum())
  plt.pie(pie_data, labels=['Null', 'Non Null'], autopct='%1.1f%%',)
  plt.show()


  # To analyse if any invoice intersects with the null/notnull CustomerID - Inference: None

  # In[221]:


  intersect = pd.Series(np.intersect1d(CustomerID_null['InvoiceNo'].values, CustomerID_notnull['InvoiceNo'].values))
  print(intersect.values)


df["CustomerID"].fillna(df["InvoiceNo"], inplace=True)

# ## Model Training
# ### 1. Modifying categorical features
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
country_one_hot = encoder.fit_transform(df['Country'])
Y_onehot=country_one_hot

# In[225]:
from sklearn.preprocessing import LabelEncoder
y_en = LabelEncoder()
Y_list = y_en.fit_transform(df["Country"])

def listcountry():
  from sklearn.preprocessing import LabelEncoder
  return_en = LabelEncoder()
  return_list = return_en.fit_transform(df["Country"])
  return return_en


from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
df["Description_Code"] = en.fit_transform(df["Description"])

def getdescription():
    d={}
    for i in range(len(df['Description'])):
        key=df['Description_Code'].iloc[i].item()
        d[df['Description'].iloc[i]] = key
    return d

#Creating X_train dataset
dataset = {'Description': df['Description_Code'], 'Quantity': df['Quantity'], 'Unit Price': df['UnitPrice'], 'Month': df['Month'], 'Day': df['Day'], 'Hour': df['Hour']}

X=pd.DataFrame(dataset)

def machinelearning():
  # #### Implementing Supervised Machine Learning Classification Algorithms

  # In[60]:


  from sklearn import metrics
  from sklearn.model_selection import train_test_split, GridSearchCV, KFold
  from sklearn.datasets import make_classification
  from sklearn.tree import DecisionTreeClassifier 
  from sklearn.svm import SVC, LinearSVC
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB
  from sklearn.linear_model import LogisticRegression, SGDClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.multiclass import OneVsRestClassifier
  import time


  # In[61]:


  #Splitting the dataset: 75% train, 25% test
  X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(X, Y_onehot, random_state = 0)
  X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(X, Y_list, random_state = 0)


  # In[62]:


  print(X_train_onehot.shape)
  print(y_train_onehot.shape)
  print(X_test_onehot.shape)
  print(y_test_onehot.shape)
  print(X_train_list.shape)
  print(y_train_list.shape)
  print(X_test_list.shape)
  print(y_test_list.shape)


  # In[63]:


  #Implementing Stochastic Gradient Descent and Logistic Regression
  numFolds = 5
  kf = KFold(numFolds, shuffle=True, random_state=0)

  Models = [LogisticRegression, SGDClassifier]
  params = [{}, {"loss": "log", "penalty": "l2", 'n_iter':100}]

  i=0
  plot_accuracy=[[],[]]
  for param, Model in zip(params, Models):
      total = 0
      for train_index, test_index in kf.split(X):
          X_train, X_test = X.iloc[train_index], X.iloc[test_index]
          y_train, y_test = Y_list[train_index], Y_list[test_index]
          reg = Model(**param)
          reg.fit(X_train, y_train)
          predictions = reg.predict(X_test)
          accuracy = metrics.accuracy_score(y_test, predictions)
          print(accuracy)
          total+=accuracy
          plot_accuracy[i].append(accuracy)
      accuracy = total / numFolds
      print("Accuracy score of {0}: {1}".format(Model.__name__, accuracy))
      i=i+1


  # In[64]:


  numFolds=[1,2,3,4,5]
  plt.plot(numFolds, plot_accuracy[0], label = "Logistic Regression")
  plt.plot(numFolds, plot_accuracy[1], label = "SGDClassifier")
  plt.ylabel("Accuracy")
  plt.xlabel("Iterations")
  plt.ylim((0.90,0.92))
  plt.legend()


  # In[69]:


  from sklearn.metrics import roc_curve, auc
  def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
      lb = LabelBinarizer()
      lb.fit(y_test)
      y_test = lb.transform(y_test)
      y_pred = lb.transform(y_pred)
      return metrics.roc_auc_score(y_test, y_pred, average=average)


  # In[70]:


  auc_roc = multiclass_roc_auc_score(y_test, predictions)
  print(auc_roc)


  # In[67]:


  testaccuracy=[]
  neighbours=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,50]
  for value in neighbours:
      start = time.time()
      knn_model = KNeighborsClassifier(n_neighbors = value)
      knn_model.fit(X_train_onehot, y_train_onehot)
      y_knn_pred = knn_model.predict(X_test_onehot)
      y_classes_pred = y_knn_pred.argmax(axis=-1)
      y_classes_test = y_test_onehot.argmax(axis=-1)
      knn_test_accuracy = metrics.accuracy_score(y_classes_test, y_classes_pred)
      testaccuracy.append(knn_test_accuracy)
      end = time.time()
      print(value, " --> ", knn_test_accuracy, " --> ", end-start)


  # In[68]:


  # Representing K_NN
  plt.plot(neighbours, testaccuracy, label="test accuarcy")
  plt.ylabel("Accuracy")
  plt.xlabel("n_neighbours")
  plt.legend()


  # In[69]:


  #Decision Tree
  testaccuracy=[]
  depth=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30]

  for value in depth:
      start = time.time()
      decision_model = DecisionTreeClassifier(max_depth = value)
      decision_model.fit(X_train_onehot, y_train_onehot) 
      y_decision_pred = decision_model.predict(X_test_onehot)
      y_classes_pred = y_decision_pred.argmax(axis=-1)
      y_classes_test = y_test_onehot.argmax(axis=-1)
      decision_test_accuracy = metrics.accuracy_score(y_classes_test, y_classes_pred)
      testaccuracy.append(decision_test_accuracy)
      end = time.time()
      print(value, " --> ", decision_test_accuracy, " --> ", end-start)


  # In[70]:


  #Representing Decision Tree
  plt.plot(depth, testaccuracy, label="test accuracy")
  plt.ylabel("Accuracy")
  plt.xlabel("Depth of the tree")
  plt.ylim((0.88,0.92))
  plt.legend()


  # In[68]:


  #Naive-Bayes
  #Since Gaussian NB does not accept any parameters there is no hyperparameter tuning
  nb = GaussianNB()
  nb_model = nb.fit(X_train_list, y_train_list)
  y_nb_pred = nb_model.predict_proba(X_test_list)
  y_classes_pred = y_nb_pred.argmax(axis=-1)
  nb_test_accuracy = metrics.accuracy_score(y_test_list, y_classes_pred)
  print("Accuracy of Naive Bayes: ", nb_test_accuracy)


  # In[ ]:


  from sklearn.metrics import confusion_matrix
  nb_cm = confusion_matrix(y_test_list, y_classes_pred)
  print(nb_cm)


  # In[ ]:


  #Alternate linear SVM
  model = LinearSVC()
  model.fit(X_train_list, y_train_list)
  y_pred = model.predict(X_test_list)


  # In[76]:


  print(metrics.accuracy_score(y_test_list, y_pred))


  # In[68]:


  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier()
  params_rf = {'n_estimators': [50, 100, 200]}
  rf_gs = GridSearchCV(rf, params_rf, cv=5)
  rf_gs.fit(X_train_onehot, y_train_onehot)
  rf_best = rf_gs.best_estimator_
  print(rf_gs.best_params_)

