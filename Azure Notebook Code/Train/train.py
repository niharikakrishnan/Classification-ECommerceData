import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nltk
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()
data_folder = args.data_folder

from azureml.core import Run
run = Run.get_context()

# Load comma separated value dataset. File should be present in the same directory else path must be sent as an argument
filepath = os.path.join(data_folder, 'data.csv')
df = pd.read_csv(filepath, encoding = 'ISO-8859-1')

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

# Removing the outliers, that is the items which have more than 50000 or less than 50000 as quantity.
df.drop([61619, 61624, 540421, 540422], inplace=True)

# Removing the outliers
df.drop([4287, 74614, 115818, 225528, 225529, 225530, 502122], inplace=True)

# Drop outliers since they negatively affect the model
df.drop([299983, 299984], inplace=True)

#Hence removed the outliers
df.drop([222681, 524602, 43702, 43703], inplace=True)

# Handle incorrect Description and remove them from the dataset
df = df[df["Description"].str.startswith("?") == False]
df = df[df["Description"].str.isupper() == True]
df = df[df["Description"].str.contains("LOST") == False]

df["CustomerID"].fillna(df["InvoiceNo"], inplace=True)

# Model Training

# 1. Modifying categorical features
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
country_one_hot = encoder.fit_transform(df['Country'])
Y_onehot=country_one_hot

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
df["Description_Code"] = en.fit_transform(df["Description"])

#Creating X_train dataset
d = {'Description': df['Description_Code'], 'Quantity': df['Quantity'], 'Unit Price': df['UnitPrice'], 'Month': df['Month'], 'Day': df['Day'], 'Hour': df['Hour']}

X=pd.DataFrame(d)

#### Implementing Supervised Machine Learning Classification Algorithms
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

#Splitting the dataset: 75% train, 25% test
X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(X, Y_onehot, random_state = 0)

#Decision Tree
decision_model = DecisionTreeClassifier(max_depth=7)
decision_model.fit(X_train_onehot, y_train_onehot) 
y_decision_pred = decision_model.predict(X_test_onehot)
y_classes_pred = y_decision_pred.argmax(axis=-1)
y_classes_test = y_test_onehot.argmax(axis=-1)
decision_test_accuracy = metrics.accuracy_score(y_classes_test, y_classes_pred)
print(decision_test_accuracy)

run.log('Accuracy', decision_test_accuracy)

from sklearn.externals import joblib
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=decision_model, filename='outputs/test_model.pkl')
run.complete()