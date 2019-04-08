# Classification-ECommerceData

Please find below the link for azure notebooks:

https://notebooks.azure.com/niharikakrishnan/projects/classification-ecommerce

#### Week 1
1. Started Andrew Ng course - Completed till Week 2
2. Read about classification problems, multi-class classification w.r.t e-commerce dataset
3. Removed all Nan Values, duplicates in the dataset (without analysing)
4. Added a new column "Total Price" for analysis of amount spent by each customer/country
5. Used matplotlib for plotting analysis of total purchase vs country, mean amount vs country, total orders vs country 

#### Week 2
1. Started with Descriptive Statistics Course - Completed till Week 3. Helped to understand various types of data and plots available to draw inferences
2. Read about multi-classification using K-NN, Decision Tree, Naive Bayes, SVM
3. Implemented Seaborn plots based on examples from descriptive statistics course
4. Outliers detected and removed 

#### Week 3
1. Converted TimeStamp to DateTime followed by separating Year, Month, Day and Hour for bucketizing
2. Read about correlation techniques - Pearson, Spearman, Kendall. Plotted the same - cmap
Inference from Pearson - Positive correlation between CutomerID, Month, Quantity and Country, Negative correlation w.r.t Description (Probable due to numerous cat.codes) 
3. Reading about if Chi-Sqaure, ANOVA tests can give inference
4. After correlation with and without null values of CustomerID - Better to retain null CustomerID values since higher correlation
5. 25% null customer ID but only 5.2 % items purchased and 14.92% amount spent by null customerID customers.
6. For data imputing, if in one invoice number, we have null and not null customerID, then it can be manipulated. However, intersection gives an empty list. Thus, cannot append values to null CustomerID
7. Read about Azure ML Services - https://www.youtube.com/watch?v=Eb7kyOJe5Kc

#### Week 4
1. Performed feature wise exploratory data analysis using matplotlib and seaborn
2. Removed the outliers in quantity and unit price - Most of the outliers were resulting in a skewed dataset hence decreasing the accuracy of the model
3. One hot encoding vs label binarizer of the predicted label (Y) since it is categorgical
4. Label encoding of description feature - Total unique description is 4700+
5. Implemented Random forest and logistic regression and checked how the model accuracy varies with a change in the parameters

#### Week 5
1. Hyperparameter Tuning of ML algorithms implemented in Week 3 & 4
2. Implemented Stochastic Gradient Descent algorithm along with Logistic Regression
3. Read about Random Search vs Grid Search for hyperparameter tuning
4. Cross Validation Techniques explored and implemented - K-Fold, Leave One-out. LOOCV was time consuming due to huge volume of dataset
5. Individual Exploratory Data Analysis of features in the X_train dataset

#### Week 6
1. Created Azure ML Workspace - Containers, Storage Account, etc
2. Free 30 days trial starts for Azure Machine Learning Service 
3. Preparation of TCS Xplore Mid Sem Assessment on Mar 2

#### Week 7
1. Creation of workspace, experiments, trial runs, metrics
2. Performed simple experiments (using scikit-learn) to check how metrics are logged
3. Azure Blob Storage vs Azure File Storage vs Block Blob vs AppendBlob - Proceeded with Blob storage
4. Lot of deprecated packages in Azure - versions differ while importing Azure ML SDK to local python code 
Hence, took decision to transfer the code to Azure Notebooks (Consists of pre-installed Azure ML packages)
5. Data can be retrieved from cloud using - Blob storage vs URL
6. Latest - Data.csv stored in blob and accessed

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

#### Week 9
1. Estimator created - Submission of run. 
2. ML Model trained within notebook using Compute Target. Run successful after 15-20 failed runs.
3. ML Model registered. Training cluster deleted
4. Scoring script created - COnsists of functions that get called from the frontend
5. Create environment file - Compiled but issue major issue while deploying (Error in Azure tutorial - Resolved next week)
6. AKS didn't allow to create VM with the required configuration. Shifted to Azure Container Instances.

#### Week 10
1. Raised an issue with Azure support since the technical chat support representative couldn't rectify the error.
Status Update: Resolved
https://social.msdn.microsoft.com/Forums/en-US/e27b3da2-eb91-4116-be99-0b88a41a1bc5/image-creation-fails-while-using-azure-container-instances-in-azure-machine-leaning-service?forum=MachineLearning
2. Conda_dependencies.yml format was modified so that it can be read without errors. 
3. Image regsitered and scoring_uri created.
4. Major issues while calling score.py file. Made neccessary changes to file - Loads json data, converts description to correct code, extracts necessary values from time stamp field converted to X_train dataframe and sent as input to ML url as input.
5. Output prediction received resulting in successful API generation.
6. Machine Learning URL can be called from local machine, remote machine, different cluser etc.


#### Week 11 & 12
1. Built basic front end using HTML and CSS
2. Learnt basices of flask to connect the HTML code with Python backend
3. Flask script created that takes user input and calls the ML url. 
4. Successfully hit the ML url to receive the prediction of a customer's origin based on user input.
