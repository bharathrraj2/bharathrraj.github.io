
---
title: "Breast Cancer Classification : Part 2 : Building Models"
date: 2020-02-12
tags: [data science, classification, machine learning]
header:
  image: "/images/breast_cancer_2/cancer.jpg"
excerpt: "data science, classification, machine learning"
mathjax: "true"
---

# CASE STUDY : BREAST CANCER CLASSIFICATION

Predicting if the cancer diagnosis is benign or malignant based on several observations/features
30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569
Class Distribution: 212 Malignant, 357 Benign
Target class:
   - Malignant
   - Benign
   
# Breast Cancer : Part 2 : Building Models


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns', None)
%config IPCompleter.greedy=True
```


```python
from sklearn.datasets import load_breast_cancer
data_load = load_breast_cancer()
data = pd.DataFrame(data_load['data'],columns=data_load['feature_names'])
data['target'] = data_load['target']
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>radius error</th>
      <th>texture error</th>
      <th>perimeter error</th>
      <th>area error</th>
      <th>smoothness error</th>
      <th>compactness error</th>
      <th>concavity error</th>
      <th>concave points error</th>
      <th>symmetry error</th>
      <th>fractal dimension error</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Train Test Split


```python
# for the model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import discretisers as dsc
from feature_engine import categorical_encoders as ce
```


```python
cont_vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
```


```python
X_train,X_test,y_train,y_test = train_test_split(data[cont_vars],data['target'],test_size=0.2, random_state=0)
```


```python

```

# Basic Model Building


```python
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

    /anaconda3/lib/python3.7/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /anaconda3/lib/python3.7/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
pipeline = Pipeline([
     ('standard_scale',StandardScaler())
])
```


```python
X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
y_pred = model.predict(X_test)
```


```python
accuracy_score(y_pred,y_test)
```




    0.9824561403508771




```python
print(classification_report(y_test, y_pred))
```

                 precision    recall  f1-score   support
    
              0       1.00      0.96      0.98        47
              1       0.97      1.00      0.99        67
    
    avg / total       0.98      0.98      0.98       114
    



```python
confusion_matrix(y_test,y_pred)
```




    array([[45,  2],
           [ 0, 67]])




```python

```

# SVC Model Tuning


```python
param_test1 = {
 'C':[0.1,1,10]
}

gsearch1 = GridSearchCV(estimator =  SVC(), 
param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=4, verbose=10)


gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

    Fitting 4 folds for each of 3 candidates, totalling 12 fits
    [CV] C=0.1 ...........................................................
    [CV] C=0.1 ...........................................................
    [CV] C=0.1 ...........................................................
    [CV] C=0.1 ...........................................................
    [CV] .................................. C=0.1, score=0.947368 -   0.0s
    [CV] .................................. C=0.1, score=0.895652 -   0.0s
    [CV] C=1 .............................................................
    [CV] .................................. C=0.1, score=0.938053 -   0.0s
    [CV] C=1 .............................................................
    [CV] C=1 .............................................................
    [CV] .................................. C=0.1, score=0.973451 -   0.0s
    [CV] .................................... C=1, score=0.965217 -   0.0s
    [CV] .................................... C=1, score=0.964912 -   0.0s
    [CV] C=1 .............................................................
    [CV] .................................... C=1, score=0.964602 -   0.0s
    [CV] C=10 ............................................................
    [CV] .................................... C=1, score=1.000000 -   0.0s
    [CV] ................................... C=10, score=0.973913 -   0.0s
    [CV] C=10 ............................................................
    [CV] ................................... C=10, score=0.956140 -   0.0s
    [CV] C=10 ............................................................
    [CV] ................................... C=10, score=0.955752 -   0.0s
    [CV] C=10 ............................................................
    [CV] ................................... C=10, score=0.991150 -   0.0s


    [Parallel(n_jobs=4)]: Batch computation too fast (0.0532s.) Setting batch_size=6.
    [Parallel(n_jobs=4)]: Done   3 out of  12 | elapsed:    0.1s remaining:    0.2s
    [Parallel(n_jobs=4)]: Done   5 out of  12 | elapsed:    0.1s remaining:    0.1s
    [Parallel(n_jobs=4)]: Done   7 out of  12 | elapsed:    0.1s remaining:    0.1s
    [Parallel(n_jobs=4)]: Done  12 out of  12 | elapsed:    0.2s finished





    ([mean: 0.93863, std: 0.02800, params: {'C': 0.1},
      mean: 0.97368, std: 0.01520, params: {'C': 1},
      mean: 0.96924, std: 0.01462, params: {'C': 10}],
     {'C': 1},
     0.9736828604794017)




```python
param_test1 = {
 'C':[0.1,1,10],
  'gamma': [0.01, 0.1, 1]
}

gsearch1 = GridSearchCV(estimator =  SVC(), 
param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=4, verbose=10)


gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

    Fitting 4 folds for each of 9 candidates, totalling 36 fits
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.913043 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.947368 -   0.0s
    [CV] ...................... C=0.1, gamma=0.01, score=0.946903 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ...................... C=0.1, gamma=0.01, score=0.955752 -   0.0s
    [CV] ....................... C=0.1, gamma=0.1, score=0.913043 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................... C=0.1, gamma=0.1, score=0.938596 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ....................... C=0.1, gamma=0.1, score=0.929204 -   0.0s
    [CV] ....................... C=0.1, gamma=0.1, score=0.911504 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] C=10, gamma=1 ...................................................
    [CV] ......................... C=0.1, gamma=1, score=0.634783 -   0.0s
    [CV] ....................... C=10, gamma=0.01, score=0.991304 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] C=10, gamma=0.01 ................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ....................... C=10, gamma=0.01, score=0.964912 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .......................... C=10, gamma=1, score=0.643478 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ......................... C=0.1, gamma=1, score=0.640351 -   0.0s
    [CV] ....................... C=10, gamma=0.01, score=0.982301 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] C=10, gamma=0.01 ................................................
    [CV] ......................... C=1, gamma=0.1, score=0.956522 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ....................... C=10, gamma=0.01, score=1.000000 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ......................... C=1, gamma=0.1, score=0.929825 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] .......................... C=10, gamma=1, score=0.640351 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ........................ C=10, gamma=0.1, score=0.956522 -   0.0s
    [CV] ......................... C=0.1, gamma=1, score=0.637168 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................... C=1, gamma=0.1, score=0.955752 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ........................ C=10, gamma=0.1, score=0.938596 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ......................... C=1, gamma=0.1, score=0.964602 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=10, gamma=0.1, score=0.929204 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ......................... C=0.1, gamma=1, score=0.637168 -   0.0s
    [CV] ........................ C=10, gamma=0.1, score=0.964602 -   0.0s
    [CV] .......................... C=10, gamma=1, score=0.646018 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] C=10, gamma=1 ...................................................
    [CV] ........................... C=1, gamma=1, score=0.643478 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=0.01, score=0.947826 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.964912 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] .......................... C=10, gamma=1, score=0.637168 -   0.0s


    [Parallel(n_jobs=4)]: Batch computation too fast (0.0442s.) Setting batch_size=8.
    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    0.1s


    [CV] ........................... C=1, gamma=1, score=0.640351 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=0.01, score=0.955752 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................ C=1, gamma=0.01, score=0.991150 -   0.0s
    [CV] ........................... C=1, gamma=1, score=0.646018 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................... C=1, gamma=1, score=0.637168 -   0.0s


    [Parallel(n_jobs=4)]: Done  36 out of  36 | elapsed:    0.3s finished





    ([mean: 0.94077, std: 0.01639, params: {'C': 0.1, 'gamma': 0.01},
      mean: 0.92309, std: 0.01132, params: {'C': 0.1, 'gamma': 0.1},
      mean: 0.63737, std: 0.00198, params: {'C': 0.1, 'gamma': 1},
      mean: 0.96491, std: 0.01631, params: {'C': 1, 'gamma': 0.01},
      mean: 0.95168, std: 0.01308, params: {'C': 1, 'gamma': 0.1},
      mean: 0.64175, std: 0.00332, params: {'C': 1, 'gamma': 1},
      mean: 0.98463, std: 0.01299, params: {'C': 10, 'gamma': 0.01},
      mean: 0.94723, std: 0.01403, params: {'C': 10, 'gamma': 0.1},
      mean: 0.64175, std: 0.00332, params: {'C': 10, 'gamma': 1}],
     {'C': 10, 'gamma': 0.01},
     0.9846293783708984)




```python
model = SVC(C=10,gamma=0.01)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)

```




    0.9824561403508771




```python
confusion_matrix(y_test,y_pred)
```




    array([[46,  1],
           [ 1, 66]])




```python

```

## XGB Classifier


```python
param_test1 = {
 'learning_rate' : [0.05,0.1],
  'n_estimators' : [100,500,800],
   'max_depth':[5]
}

gsearch1 = GridSearchCV(estimator =  XGBClassifier(
 subsample=0.9,
 colsample_bytree=0.9,
 objective = 'binary:logistic',
 nthread=4,
 seed=0),
                        
param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=4, verbose=10)


gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




```

    Fitting 4 folds for each of 6 candidates, totalling 24 fits
    [CV] learning_rate=0.05, max_depth=5, n_estimators=100 ...............
    [CV] learning_rate=0.05, max_depth=5, n_estimators=100 ...............
    [CV] learning_rate=0.05, max_depth=5, n_estimators=100 ...............
    [CV] learning_rate=0.05, max_depth=5, n_estimators=100 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=100, score=0.929825 -   0.2s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=500 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=100, score=0.964602 -   0.2s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=500 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=100, score=0.982301 -   0.2s
    [CV]  learning_rate=0.05, max_depth=5, n_estimators=100, score=0.956522 -   0.2s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=500 ...............
    [CV] learning_rate=0.05, max_depth=5, n_estimators=500 ...............


    [Parallel(n_jobs=4)]: Batch computation too fast (0.1960s.) Setting batch_size=2.
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=500, score=0.973451 -   0.4s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=800 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=500, score=0.973913 -   0.6s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=800 ...............


    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    0.6s
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=500, score=0.973451 -   0.7s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=100 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=500, score=0.929825 -   0.7s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=100 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=100, score=0.973451 -   0.3s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=100 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=100, score=0.956522 -   0.3s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=100 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=100, score=0.938596 -   0.2s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=500 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=100, score=0.991150 -   0.3s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=500 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=800, score=0.973913 -   1.1s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=800 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=800, score=0.973451 -   1.1s
    [CV] learning_rate=0.05, max_depth=5, n_estimators=800 ...............


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=500, score=0.973451 -   0.5s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=500 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=500, score=0.982609 -   0.5s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=500 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=800, score=0.929825 -   0.7s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=800 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=500, score=0.947368 -   0.4s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=800 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=500, score=0.991150 -   0.5s


    [Parallel(n_jobs=4)]: Done  14 out of  24 | elapsed:    2.4s remaining:    1.7s
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.05, max_depth=5, n_estimators=800, score=0.973451 -   1.1s


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=800, score=0.973913 -   0.7s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=800 ................


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=800, score=0.973451 -   0.8s
    [CV] learning_rate=0.1, max_depth=5, n_estimators=800 ................


    [Parallel(n_jobs=4)]: Done  20 out of  24 | elapsed:    3.0s remaining:    0.6s
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=800, score=0.947368 -   0.3s


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:


    [CV]  learning_rate=0.1, max_depth=5, n_estimators=800, score=0.991150 -   0.3s


    [Parallel(n_jobs=4)]: Done  24 out of  24 | elapsed:    3.5s finished





    ([mean: 0.95831, std: 0.01891, params: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100},
      mean: 0.96266, std: 0.01896, params: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 500},
      mean: 0.96266, std: 0.01896, params: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 800},
      mean: 0.96493, std: 0.01952, params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100},
      mean: 0.97364, std: 0.01641, params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500},
      mean: 0.97147, std: 0.01564, params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 800}],
     {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500},
     0.9736447216540773)




```python
model = XGBClassifier(
 learning_rate= 0.1, max_depth= 5, n_estimators=500,
 subsample=0.9,
 colsample_bytree=0.9,
 objective = 'binary:logistic',
 nthread=4,
 seed=0)

model.fit(X_train,y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.9, gamma=0,
           learning_rate=0.1, max_delta_step=0, max_depth=5,
           min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
           nthread=4, objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=0, silent=None,
           subsample=0.9, verbosity=1)




```python
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:





    0.9912280701754386




```python
confusion_matrix(y_test,y_pred)
```




    array([[47,  0],
           [ 1, 66]])




```python
ser = pd.Series(model.feature_importances_)
ser.index = cont_vars
ser = ser.sort_values(ascending=False)
ser.plot.bar(figsize=(15,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1afbaa90>




![alt]({{ site.url }}{{ site.baseurl }}/images/breast_cancer_2/output_32_1.png)



```python
accuracy_score(model.predict(X_train),y_train)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:





    1.0




```python
accuracy_score(model.predict(X_test),y_test)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:





    0.9912280701754386




```python

```


```python

```
