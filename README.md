# truck-breakdown
This project was born in the context of the 2017 Unearthed Hackathon (https://unearthed.solutions/hackathons/unearthed-vancouver-2017/), hosted in Vancouver, BC. In a team of 4, we attempted to build a solution for the following challenge:

Using data to proactively predict Equipment Failure prior to the actual failure event occurs by using equipment production, maintenance and alerts information.

# Objective
Our goal was to implement machine learning models capable of predicting the probability of engine breakdowns in haul-trucks (see picture below) for a given future time interval. To address our goal, we focused on two sets of information:

- The measurements of the chemical composition of the engine oil for each truck, taken at regular intervals. This information is useful as the presence of impurities (e.g. water, coolant fuid), or high levels of certain metals (such as copper) in the oil might indicate that parts of the engine are failing. 
- The maintenance reports, where the exact dates of failures as well as the part broken is described.

# The Models
We implemented the following models:
1) neural network with one hidden layer, using Tensorflow
2) support vector machine model, using scikit-learn
3) random forest model, using scikit-learn
4) gaussian process model, using scikit-learn

# The data
The available data 'OilAnalysis_vs_DaystoFailure.csv' include oil-composiotion information
