# Importing required modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

#Load dataset
df = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

# Display Information about the dataset

def information(df):
	print(df.describe())
	print(df.dtypes)
	print(df.head())

def shape(df):
	print(df.shape)

shape(df)

#Data Preprocessing
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")

# check missing values for each column 
print(df.isnull().sum().sort_values(ascending=False))

#No. of order canceled
order_canceled = df['InvoiceNo'].apply(lambda x:int('C' in x))
no_cancel= order_canceled.sum()
print('Number of orders canceled:' + str(no_cancel))

# Drop the rows that have both Description and CustomerID as None.
df.dropna(axis=0, subset=['Description', 'CustomerID'], inplace=True)
# Drop duplicates
df.drop_duplicates(keep='first', inplace=True)
# Drop unnecessary columns from dataframe 
df.drop(axis=1, columns='InvoiceNo', inplace=True)
shape(df)

#Count countries - 37 countries
print(df['Country'].value_counts(normalize=True))

#Appending total price of each purchase
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']

#Total order spent by country
total_orders=df.groupby('Country')['Quantity'].count().sort_values(ascending=False)
#Total amount spent by country
total_amount=df.groupby('Country')['TotalPrice'].mean().sort_values(ascending=False)
#Mean amount spent by country
mean_amount=df.groupby('Country')['TotalPrice'].mean().sort_values(ascending=False

description=df.groupby('Description').count().sort_values(ascending = False)

#Description plot
description.plot('bar')
plt.xlabel('Country')
plt.ylabel('Description')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()

#Number of Orders per country plot
total_orders.plot('bar')
plt.xlabel('Country')
plt.ylabel('Number of Orders')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()

#Total amount spent per country
total_amount.plot('bar')
plt.xlabel('Country')
plt.ylabel('Total amount')
plt.title('Total amount spent per Country', fontsize=16)
plt.show()

#Mean amount spent per country plot
mean_amount.plot('bar')
plt.xlabel('Country')
plt.ylabel('Total Purchase')
plt.title('Amount spent by every Country', fontsize=16)
plt.show()

#Extract product description
