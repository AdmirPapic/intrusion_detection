# Anomaly-based Intrusion Detection Systems: Project Overview 
* Created a tool that classifies online behavior as either BENIGN or MALIGNANT (99.95 % accuracy) to help organizations identify cyber attacks
* Selected 40 features of network traffic which have the highest importance metrics 
* Optimized, compared and utilized K Nearest Neighbors, Logistic Regression, Support Vector Classifier and Random Forest Classifier models using GridsearchCV and model performance metrics to obtain the best model 
* Built a client facing API using flask 

## Resources 
**Python Version:** 3.8.17

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle  

**For Web Framework Requirements:**  ``pip install -r requirements.txt`` 

**Dataset:** https://www.kaggle.com/datasets/cicdataset/cicids2017 

**Feature Selection:** https://towardsdatascience.com/understanding-feature-importance-and-how-to-implement-it-in-python-ff0287b20285 

**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2, 
https://www.youtube.com/watch?v=nUOh_lDMHOU&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t&index=6&pp=iAQB


## Dataset
CICIDS2017 dataset contains benign and the most up-to-date common attacks. It provides data which is the result of network traffic analysis using CICFlowMeter with flows based on time stamp, source, destionation IPs, source and destination ports, protocols and attacks. Efforts have been made towards generating realistic background traffic for this dataset.

## Data Cleaning
In order for the data to be usable to machine learning models, it had to undergo cleaning. The following changes have been made to the original dataset:
 
*	Rows with empty cells were removed
*	Indents at the begining of certain column names were removed
*	All the attacks were labeled as "MALIGNANT" behavior
*	The two remaining classes were balanced by downsampling the number of data points labeled as "BENIGN" behavior
*	Rows containing infinite values were removed

## EDA
The types of all features were either integers or floats. Their mean, minimum, 25%, medium, 75%, maximum were examined. Histograms for a number of features were examined, although the large amount of features dictated a need for feature selection. 



## Feature Selection

After splitting the data in a train and test set, it was fitted on a Random Forest Classifier model with 150 estimators (decision trees). Because of the excellent performance of the model, impurity-based feature importances were computed from that same model. 
The following bar graph displays the relative importance of each feature in the dataset.

![alt text](https://github.com/AdmirPapic/intrusion_detection/blob/master/images/feature_importances.png "Feature Importances")

The 38 least important features were dropped in order to improve future runtimes (40 features were kept). The model was fitted again on the remaining data and metrics show a negligible difference between them.

## Model Building
Four different types of models were built and their performance was compared.
They were evaluated using their accuracy score. A binary confusion matrix was plotted and examined for each of the models.

*	**K Nearest Neighbors** – The number of nearest neighbors was tuned using GridSearchCV. Optimal results were obtained for K = 3.

![alt text](https://github.com/AdmirPapic/intrusion_detection/blob/master/images/k_nearest_neighbors.png "K Nearest Neighbors")
  
*	**Logistic Regression** – From the start, this model had poor performance and a very large train set error which suggests the need for adding polynomial features. However, the number of features is already quite large and therefore it was not further considered.
*	**Support Vector Classifier** – Both Linear and Gaussian Support Vector Classifiers were fitted and yielded poor results.

![alt text](https://github.com/AdmirPapic/intrusion_detection/blob/master/images/linear_svc.png "Support Vector Classifier")

*	**Random Forest Classifier** – A random forest classifier with near perfect accuracy was fitted, consisting of 150 decision trees.

![alt text](https://github.com/AdmirPapic/intrusion_detection/blob/master/images/rf_classifier.png "Random Forest Classifier")

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest Classifier** : Best accuracy score = 99.95 %
*	**K Nearest Neighbors Classifier**: Best accuracy score = 99.82 %
*	**Support Vector Classifier**: Best accuracy score = 97.51 %
*	**Logistic Regression**: Best accuracy score = 96.95 %

## Productionization 
A flask API endpoint that was hosted on a local webserver was built by following along with the TDS tutorial in the reference section. The API endpoint takes in a request with a list of values for network traffic features and returns either the "BENIGN" or "MALIGNANT" label. 
