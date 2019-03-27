# Classification-ECommerceData

Please find below the link for azure notebooks:

https://notebooks.azure.com/niharikakrishnan/projects/classification-ecommerce

Open issue on github: https://github.com/conda/conda/issues/8180
Apparently it is a bug with Conda.


#### Week 1
1. Started Andrew Ng course - Completed till Week 2
2. Read about classification problems, multi-class classification w.r.t e-commerce dataset
3. Removed all Nan Values, duplicates in the dataset (without analysing)
4. Added a new column "Total Price" for analysis of amount spent by each customer/country
5. Used matplotlib for plotting analysis of total purchase vs country, mean amount vs country, total orders vs country 

#### Week 2
1. Started with Descriptive Statistics Course - Completed till Week 3. Helped to understand various types of data and plots available to draw inferences
2. Read about multi-classification using K-NN, Decision Tree, Naive Bayes, SVM
3. Converted Categorical values to unique codes for implementing ML models
4. Implemented Seaborn plots based on examples from descriptive statistics course

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
1. Performed feature wise exploratory data analysis
2. Removed the outliers in quantity and unit price
3. One hot encoding of predicted variable (y)
4. Label encoding of description feature
5. Implemented Random forest and logistic regression

#### Week 5
1. Hyperparameter Tuning of ML algorithms implemented in Week 3 & 4
2. Implemented SGD algorithm along with Logistic Regression
3. Grid search for hyperparameter tuning and K-Fold cross validation
4. Further EDA of X features

#### Week 6
1. Created Azure ML Workspace
2. Free 30 days trial starts for Azure ML
3. Mainly prepared for TCS Xplore Mid Sem Assessment on Mar 2

#### Week 7
1. Creation of workspace, experiments, trial runs, metrics
2. Performed simple experiments to check how metrics are logged
3. Azure Blob Storage vs Azure File Storage vs Block Blob vs AppendBlob
4. Lot of deprecated packages in Azure - versions differ. Read about each and experimented with the same. 
5. Data can also be fetched by the url inside the container of the storage account. 
6. Latest - Data.csv stored in blob and retrieved

#### Week 8
1. Transfering all code to Azure Notebook
2. File uploaded to storage account
3. Experimenting 3 types of training - Local Machine, Withink notebook, Remote Server
Local Machine - Done
Within Notebook - Done
Remote Server - In Progress
