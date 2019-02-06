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
1. Read about correlation techniques - Pearson, Kendall. Plotted the same - cmap
Inference - High correlation between CutomerID and Country
2. However, since dataset has categorical values - Chi-Square and ANOVA may be a better fit (Still reading and implementing)
3. After correlation with and without null values of CustomerID - Better to retain null CustomerID values
4. 25% null customer ID but on;y 5.2% amount spent by null customerID customers
5. For data impuning, if in one invoice number, there are fields with missing customerID, then it can be manipulated. However, intersection gives an empty list. Thus, cannot append values to null CustomerID
6. Converted TimeStamp to DateTime followed by separating Year, Month, Day and Hour for bucketizing
7. Read about Azure ML Services - https://www.youtube.com/watch?v=Eb7kyOJe5Kc
