# Multi-Class Classification of E-Commerce Data using Supervised Machine Learning Algorithms

All code available in azure notebooks:

https://notebooks.azure.com/niharikakrishnan/projects/classification-ecommerce

## Dataset
This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

## Aim
We aim to implement various classification algorithms like SVM, Logistic Regression, Naive Bayes, Random Forest, SGD, k-NN to predict a customer's origin and to compare the performance of these supervised machine learning models. To deploy the ML Model using Microsoft Azure Machine Learning Service.


## Week-wise Progress


### Data Pre-processing
#### Week 1
1. Started Andrew Ng course - Completed till Week 2
2. Read about classification problems, multi-class classification w.r.t e-commerce dataset
3. Removed all Nan Values, duplicates in the dataset (without analysing)
4. Added a new column "Total Price" for analysis of amount spent by each customer/country
5. Used matplotlib for plotting analysis of total purchase vs country, mean amount vs country, total orders vs country 

![dataset](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/dataset.png)


#### Week 2
1. Started with Descriptive Statistics Course - Completed till Week 3. Helped to understand various types of data and plots available to draw inferences
2. Read about multi-classification using K-NN, Decision Tree, Naive Bayes, SVM
3. Implemented Seaborn plots based on examples from descriptive statistics course
4. Outliers detected and removed 

![preprocess_1](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/preprocess_2.png)

#### Week 3
1. Converted TimeStamp to DateTime followed by separating Year, Month, Day and Hour for bucketizing
2. Read about correlation techniques - Pearson, Spearman, Kendall. Plotted the same - cmap
Inference from Pearson - Positive correlation between CutomerID, Month, Quantity and Country, Negative correlation w.r.t Description (Probable due to numerous cat.codes) 
3. Reading about if Chi-Sqaure, ANOVA tests can give inference
4. After correlation with and without null values of CustomerID - Better to retain null CustomerID values since higher correlation
5. 25% null customer ID but only 5.2 % items purchased and 14.92% amount spent by null customerID customers.
6. For data imputing, if in one invoice number, we have null and not null customerID, then it can be manipulated. However, intersection gives an empty list. Thus, cannot append values to null CustomerID
7. Read about Azure ML Services - https://www.youtube.com/watch?v=Eb7kyOJe5Kc

![preprocess_2](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/preprocess_1.png)


### Machine Learning Models
#### Week 4
1. Performed feature wise exploratory data analysis using matplotlib and seaborn
2. Removed the outliers in quantity and unit price - Most of the outliers were resulting in a skewed dataset hence decreasing the accuracy of the model
3. One hot encoding vs label binarizer of the predicted label (Y) since it is categorgical
4. Label encoding of description feature - Total unique description is 4700+
5. Implemented Random forest and logistic regression and checked how the model accuracy varies with a change in the parameters

``` python
  #K-NN
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
 ```
 ``` python
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
   ```

#### Week 5
1. Hyperparameter Tuning of ML algorithms implemented in Week 3 & 4
2. Implemented Stochastic Gradient Descent algorithm along with Logistic Regression
3. Read about Random Search vs Grid Search for hyperparameter tuning
4. Cross Validation Techniques explored and implemented - K-Fold, Leave One-out. LOOCV was time consuming due to huge volume of dataset
5. Individual Exploratory Data Analysis of features in the X_train dataset

``` python
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
```

#### Week 6
1. Created Azure ML Workspace - Containers, Storage Account, etc
2. Free 30 days trial starts for Azure Machine Learning Service 

### Model Deployment using Microsoft Azure Machine Learning Service
#### Week 7
1. Creation of workspace, experiments, trial runs, metrics
2. Performed simple experiments (using scikit-learn) to check how metrics are logged
3. Azure Blob Storage vs Azure File Storage vs Block Blob vs AppendBlob - Proceeded with Blob storage
4. Lot of deprecated packages in Azure - versions differ while importing Azure ML SDK to local python code 
Hence, took decision to transfer the code to Azure Notebooks (Consists of pre-installed Azure ML packages)
5. Data can be retrieved from cloud using - Blob storage vs URL
6. Latest - Data.csv stored in blob and accessed

![Azure ML](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/resource.png)

#### Week 8
1. Transfering all code to Azure Notebook
2. Compute resourcces created (Azure Virtual Machine - Standard D2 V2)
3. Dataset uploaded to blob storage account
4. Experimenting 3 types of training  the model - Local Machine, Withink notebook, Remote Server
Local Machine - Done
Within Notebook - Done
Remote Server - Done
5. Proceeded with Decision Tree Model with depth 7 as final model due to highest accuracy and least processing time
6. Created a separate train.py consisting of only necessary preprocessing and training model
7. Model saved as .pkl file

``` python
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()
data_folder = args.data_folder

from azureml.core import Run
run = Run.get_context()

# Load comma separated value dataset. File should be present in the same directory else path must be sent as an argument
filepath = os.path.join(data_folder, 'data.csv')
df = pd.read_csv(filepath, encoding = 'ISO-8859-1')

df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month 
df['Day'] = df['InvoiceDate'].dt.day 
df['Hour'] = df['InvoiceDate'].dt.hour 
df.drop(columns=['InvoiceDate'], inplace=True)
df.drop_duplicates(keep='first', inplace=True)

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
```

#### Week 9
1. Estimator created - Submission of run. 
2. ML Model trained within notebook using Compute Target. Run successful after 15-20 failed runs.
3. ML Model registered. Training cluster deleted
4. Scoring script created - Consists of functions that get called from the frontend
5. Create environment file - Compiled but issue major issue while deploying (Error in Azure tutorial - Resolved next week)
6. AKS didn't allow to create VM with the required configuration. Shifted to Azure Container Instances.

``` python
import json
import numpy as np
from sklearn.externals import joblib
from azureml.core.model import Model

# In[105]: #load the model
def init():
    global model
    model_path = Model.get_model_path('test_model')
    model = joblib.load(model_path)

def run(dictdata):
    des = {
        'WHITE HANGING HEART T-LIGHT HOLDER': 3833, 'WHITE METAL LANTERN': 3841, 'CREAM CUPID HEARTS COAT HANGER': 885, 
        'KNITTED UNION FLAG HOT WATER BOTTLE': 1850, 'RED WOOLLY HOTTIE WHITE HEART.': 2833, 'SET 7 BABUSHKA NESTING BOXES': 3080,
        'GLASS STAR FROSTED T-LIGHT HOLDER': 1474 }
    
    data = json.loads(dictdata)
    for i in range(len(data["data"])):
        key=data["data"][i]['description']
        timestamp = data["data"][i]['timestamp']
        date = timestamp.split()[0]
        time = timestamp.split()[1]
        if '/' in date:
            date=date.split('/')
            data["data"][i]["month"] = date[0] 
            data["data"][i]["day"] = date[1]
            data["data"][i]["hour"] = time.split(':')[0] 
            del data["data"][i]['timestamp']
        elif '-' in date:
            date = date.split('-')
            data["data"][i]["month"] = date[0] 
            data["data"][i]["day"] = date[1]
            data["data"][i]["hour"] = time.split(':')[0]
            del data["data"][i]['timestamp']

        if key in des:
            data["data"][i]['description']=des[key]
        else:
            print("Description not found")
            
    final_dict = {}
    for i in range(len(data["data"])):
        final_dict = data["data"][i]
        arr = np.array([final_dict[key] for key in ('description', 'quantity', 'unitprice', 'month', 'day', 'hour')]).T
        arr1 = np.reshape(arr, (-1, 6))
        y_pred = model.predict(arr1)
        y_class = y_pred.argmax(axis = -1)
    return y_class.tolist()
```

#### Week 10
1. Raised an issue with Azure support since the technical chat support representative couldn't rectify the error.
Status Update: Resolved
https://social.msdn.microsoft.com/Forums/en-US/e27b3da2-eb91-4116-be99-0b88a41a1bc5/image-creation-fails-while-using-azure-container-instances-in-azure-machine-leaning-service?forum=MachineLearning
2. Conda_dependencies.yml format was modified so that it can be read without errors. 
3. Image regsitered and scoring_uri created.
4. Major issues while calling score.py file. Made neccessary changes to file - Loads json data, converts description to correct code, extracts necessary values from time stamp field converted to X_train dataframe and sent as input to ML url as input.
5. Output prediction received resulting in successful API generation.
6. Machine Learning URL can be called from local machine, remote machine, different cluser etc.

![aci_1](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/aci_1.png)

![aci_2](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Azure%20Portal%20Images/aci_2.png)


### Flask and Front-end
#### Week 11 & 12
1. Built basic front end using HTML and CSS
2. Learnt basics of flask to connect the HTML code with Python backend
3. Flask script created that takes user input and calls the ML url. 
4. Successfully hit the ML url to receive the prediction of a customer's origin based on user input.

``` python 
app = Flask(__name__)

# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

# prediction function
def ValuePredictor(jsondata):
	headers = {'Content-Type': 'application/json'}
	uri = "http://20.185.111.89:80/score"
	resp = requests.post(uri, jsondata, headers = headers)
	finalclass = int(resp.text.strip('[]'))
	return finalclass

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
    	list1=[]
    	description = request.form['description']
    	quantity = request.form['quantity']
    	unitprice = request.form['unitprice']
    	timestamp = request.form['timestamp']
    	dictionary = {"description": description, "quantity": quantity, "unitprice": unitprice, "timestamp": timestamp}
    	list1.append(dictionary)
    	final_dict = {"data": list1}
    	jsondata = json.dumps(final_dict)
    	result = ValuePredictor(jsondata)
    	country = {
      36: 'United Kingdom', 13: 'France', 0: 'Australia', 24: 'Netherlands', 14: 'Germany', 25: 'Norway', 10: 'EIRE', 
      33: 'Switzerland', 31: 'Spain', 26: 'Poland', 27: 'Portugal', 19: 'Italy', 3: 'Belgium', 22: 'Lithuania', 20: 'Japan', 
      17: 'Iceland', 6: 'Channel Islands', 9: 'Denmark', 7: 'Cyprus', 32: 'Sweden', 1: 'Austria', 18: 'Israel', 12: 'Finland', 
      2: 'Bahrain', 15: 'Greece', 16: 'Hong Kong', 30: 'Singapore', 21: 'Lebanon', 35: 'United Arab Emirates', 29: 'Saudi Arabia', 
      8: 'Czech Republic', 5: 'Canada', 37: 'Unspecified', 4: 'Brazil', 34: 'USA', 11: 'European Community', 23: 'Malta', 28: 'RSA'}

    	if result in country:
    		prediction = country[result]
    		return render_template("index.html", prediction = prediction)
    	else:
    		prediction = "Not found"
    		return render_template("index.html", prediction = prediction)

if __name__ == '__main__':
	app.run(debug = True, threaded=True) 
  ```

## Output Images

![output](https://github.com/niharikakrishnan/Classification-ECommerceData/blob/master/Web%20Frontend%20Images/output.png)


## Issues
1. Dataset very skewed for Customer Origin Classification Problem - 95% data point belong to United Kingdom

*Solution:* Sampling
 - Simple Random Sampling
 - Stratified Sampling
 - Cluster Sampling
 - Multistage Sampling
 
2. Azure Kubernetes Service couldn't be implmented in Free Trial subscription. Hence shifted to Azure Container Instances

## Online Courses

| Course Name  | Status |
| ------------- | ------------- |
| Descriptive Statistics| Audited: Week 4 / Week 5 |
| Inferential Statistics  | Audited: Week 4 / Week 5  |
| Machine Learning by Andrew NG  | Audited: Week 5 / Week 11  |

## External References
1. Stack Overflow
2. Github Issues
3. Packages documentations / Microsoft Azure Docs & Tutorials
4. Azure Technical Support
5. Coursera / YouTube videos
6. Towards Data Science / Medium / Analytics Vidhya
7. Kaggle
