# truck-breakdown
This project was born in the context of the 2017 Unearthed Hackathon (https://unearthed.solutions/hackathons/unearthed-vancouver-2017/), hosted in Vancouver, BC. In a team of 4, we attempted to build a solution for the following challenge:

Using data to proactively predict Equipment Failure prior to the actual failure event occurs by using equipment production, maintenance and alerts information. 

The challenge and data was provided by SSR Mining.

# Objective
Our goal was to implement machine learning models capable of predicting the probability of engine breakdowns in haul-trucks (see picture below, credit = Wikipedia) for a given future time interval. To address our goal, we focused on two sets of information:

- The measurements of the chemical composition of the engine oil for each truck, taken at regular intervals. This information is useful as the presence of impurities (e.g. water, coolant fuid), or high levels of certain metals (such as copper) in the oil might indicate that parts of the engine are failing. 
- The maintenance reports, where the exact dates of failures as well as the part broken is described.

![haul-truck](https://upload.wikimedia.org/wikipedia/commons/5/5c/CamionFermont.png?raw=true "haul-truck")

# The Models
We implemented the following models:
1) neural network with one hidden layer, using Tensorflow
2) support vector machine model, using scikit-learn
3) random forest model, using scikit-learn
4) gaussian process model, using scikit-learn

# The Data
The available data 'OilAnalysis_vs_DaystoFailure.csv' contains the following information from 12 haul trucks acquired over 8 months:
- Truck ID
- compart -> Refers to the compartment of the truck for which the oil was analysed (e.g. engine, hydraulic, etc.). We only focused on engine oil.
- oilhours -> age of the oil, in hours. This is useful to identify when new oil was added.
- A series of measurements output, including V100 (viscosity of oil at 100 C), Fe, Cu, Pb, ... (presence of these elements in oil, in ppm). In our study, we performed a pre-liminary analysis of the distribution of each of these feautres that could be potentially predictors of engine breakdown. We chose the following 7 features of the engine oil measurements: oilhours, V100, Fe, Cu, Al, Mo, Sulf. 
-DamageDelta -> represents the time interval (in days) from the date of oil measurement until the next engine breakdown. 

Please note that the datafile provided in this repository is the result of our exhaustive analysis and processing of the initial data given by the challenge organizers. It summarizes the features that we thought were more promising to predict engine breakdown of trucks using engine oil composition analysis.

# The Results
The two figures below show the sensitivity and specificity of each of the investigated models to predict truck breakdowns as a function of the time window. Overall, we could not confidently say that any of our models can make meaningful predictions. We think that this poor performance was due to the limited number of training examples (70 data points in total), as well as the lack of expertise to ensure that the engine failures included in our model really changed the oil composition.

![sensitivity](https://github.com/plesqui/truck-breakdown/blob/master/performance-sensitivity.png?raw=true "Sensitivity")
![specificity](https://github.com/plesqui/truck-breakdown/blob/master/performance_specificity.png?raw=true "Specificity")
