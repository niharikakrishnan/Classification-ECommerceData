# Classification-ECommerceData

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
3. After correlation with and without null values of CustomerID - Better to retain null CustomerID values since higher correlation
4. 25% null customer ID but only 5.2 % items purchased and 14.92% amount spent by null customerID customers.
5. For data imputing, if in one invoice number, we have null and not null customerID, then it can be manipulated. However, intersection gives an empty list. Thus, cannot append values to null CustomerID
6. Read about Azure ML Services - https://www.youtube.com/watch?v=Eb7kyOJe5Kc
