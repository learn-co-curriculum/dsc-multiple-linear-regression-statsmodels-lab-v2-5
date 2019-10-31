
# Multiple Linear Regression in Statsmodels - Lab

## Introduction
In this lab, you'll practice fitting a multiple linear regression model on the Boston Housing dataset!

## Objectives
You will be able to:
* Determine if it is necessary to perform normalization/standardization for a specific model or set of data
* Use standardization/normalization on features of a dataset
* Identify if it is necessary to perform log transformations on a set of features
* Perform log transformations on different features of a dataset
* Use statsmodels to fit a multiple linear regression model
* Evaluate a linear regression model by using statistical performance metrics pertaining to overall model and specific parameters


## The Boston Housing Data

We pre-processed the Boston Housing data again. This time, however, we did things slightly different:
- We dropped `'ZN'` and `'NOX'` completely 
- We categorized `'RAD'` in 3 bins and `'TAX'` in 4 bins
- We transformed `'RAD'` and `'TAX'` to dummy variables and dropped the first variable to eliminate multicollinearity
- We used min-max-scaling on `'B'`, `'CRIM'`, and `'DIS'` (and log transformed all of them first, except `'B'`)
- We used standardization on `'AGE'`, `'INDUS'`, `'LSTAT'`, and `'PTRATIO'` (and log transformed all of them first, except for `'AGE'`) 


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_features = boston_features.drop(['NOX', 'ZN'],axis=1)

# First, create bins for based on the values observed. 3 values will result in 2 bins
bins = [0, 6,  24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# First, create bins for based on the values observed. 4 values will result in 3 bins
bins = [0, 270, 360, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix='TAX', drop_first=True)
rad_dummy = pd.get_dummies(bins_rad, prefix='RAD', drop_first=True)
boston_features = boston_features.drop(['RAD', 'TAX'], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)
```


```python
age = boston_features['AGE']
b = boston_features['B']
logcrim = np.log(boston_features['CRIM'])
logdis = np.log(boston_features['DIS'])
logindus = np.log(boston_features['INDUS'])
loglstat = np.log(boston_features['LSTAT'])
logptratio = np.log(boston_features['PTRATIO'])

# Min-Max scaling
boston_features['B'] = (b-min(b))/(max(b)-min(b))
boston_features['CRIM'] = (logcrim-min(logcrim))/(max(logcrim)-min(logcrim))
boston_features['DIS'] = (logdis-min(logdis))/(max(logdis)-min(logdis))

# Standardization
boston_features['AGE'] = (age-np.mean(age))/np.sqrt(np.var(age))
boston_features['INDUS'] = (logindus-np.mean(logindus))/np.sqrt(np.var(logindus))
boston_features['LSTAT'] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
boston_features['PTRATIO'] = (logptratio-np.mean(logptratio))/(np.sqrt(np.var(logptratio)))
```


```python
boston_features.head()
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
      <th>CRIM</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>RAD_(6, 24]</th>
      <th>TAX_(270, 360]</th>
      <th>TAX_(360, 712]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-1.704344</td>
      <td>0.0</td>
      <td>6.575</td>
      <td>-0.120013</td>
      <td>0.542096</td>
      <td>-1.443977</td>
      <td>1.000000</td>
      <td>-1.275260</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.153211</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>6.421</td>
      <td>0.367166</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>1.000000</td>
      <td>-0.263711</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.153134</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>7.185</td>
      <td>-0.265812</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>0.989737</td>
      <td>-1.627858</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.171005</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>6.998</td>
      <td>-0.809889</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>0.994276</td>
      <td>-2.153192</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.250315</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>7.147</td>
      <td>-0.511180</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>1.000000</td>
      <td>-1.162114</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Run a linear model in statsmodels


```python
X = boston_features
y = pd.DataFrame(boston.target, columns = ['price'])
```


```python
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.779</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.774</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   144.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 21 Aug 2019</td> <th>  Prob (F-statistic):</th> <td>5.08e-153</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:53:00</td>     <th>  Log-Likelihood:    </th> <td> -1458.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2942.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   493</td>      <th>  BIC:               </th> <td>   2997.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    12</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>          <td>    8.6442</td> <td>    3.189</td> <td>    2.711</td> <td> 0.007</td> <td>    2.379</td> <td>   14.910</td>
</tr>
<tr>
  <th>CRIM</th>           <td>   -1.9538</td> <td>    2.115</td> <td>   -0.924</td> <td> 0.356</td> <td>   -6.110</td> <td>    2.202</td>
</tr>
<tr>
  <th>INDUS</th>          <td>   -0.8046</td> <td>    0.362</td> <td>   -2.220</td> <td> 0.027</td> <td>   -1.517</td> <td>   -0.093</td>
</tr>
<tr>
  <th>CHAS</th>           <td>    2.5959</td> <td>    0.796</td> <td>    3.260</td> <td> 0.001</td> <td>    1.032</td> <td>    4.160</td>
</tr>
<tr>
  <th>RM</th>             <td>    2.6466</td> <td>    0.408</td> <td>    6.488</td> <td> 0.000</td> <td>    1.845</td> <td>    3.448</td>
</tr>
<tr>
  <th>AGE</th>            <td>    0.0794</td> <td>    0.352</td> <td>    0.226</td> <td> 0.821</td> <td>   -0.612</td> <td>    0.770</td>
</tr>
<tr>
  <th>DIS</th>            <td>  -10.0962</td> <td>    1.856</td> <td>   -5.439</td> <td> 0.000</td> <td>  -13.743</td> <td>   -6.449</td>
</tr>
<tr>
  <th>PTRATIO</th>        <td>   -1.4867</td> <td>    0.241</td> <td>   -6.160</td> <td> 0.000</td> <td>   -1.961</td> <td>   -1.013</td>
</tr>
<tr>
  <th>B</th>              <td>    3.8412</td> <td>    0.986</td> <td>    3.897</td> <td> 0.000</td> <td>    1.905</td> <td>    5.778</td>
</tr>
<tr>
  <th>LSTAT</th>          <td>   -5.6288</td> <td>    0.354</td> <td>  -15.912</td> <td> 0.000</td> <td>   -6.324</td> <td>   -4.934</td>
</tr>
<tr>
  <th>RAD_(6, 24]</th>    <td>    1.3380</td> <td>    0.672</td> <td>    1.990</td> <td> 0.047</td> <td>    0.017</td> <td>    2.659</td>
</tr>
<tr>
  <th>TAX_(270, 360]</th> <td>   -1.2598</td> <td>    0.600</td> <td>   -2.100</td> <td> 0.036</td> <td>   -2.438</td> <td>   -0.081</td>
</tr>
<tr>
  <th>TAX_(360, 712]</th> <td>   -2.1461</td> <td>    0.704</td> <td>   -3.047</td> <td> 0.002</td> <td>   -3.530</td> <td>   -0.762</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>106.730</td> <th>  Durbin-Watson:     </th> <td>   1.093</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 432.101</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.891</td>  <th>  Prob(JB):          </th> <td>1.48e-94</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.162</td>  <th>  Cond. No.          </th> <td>    117.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Run the same model in scikit-learn


```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# coefficients
linreg.coef_
```




    array([[ -1.95380233,  -0.80457549,   2.59586776,   2.64657111,
              0.07939727, -10.09618465,  -1.48666599,   3.8412139 ,
             -5.62879369,   1.33796317,  -1.25977612,  -2.14606188]])




```python
# intercept
linreg.intercept_
```




    array([8.64415614])



## Interpret the coefficients for PTRATIO, PTRATIO, LSTAT

- CRIM: per capita crime rate by town
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population

## Predict the house price given the following characteristics (before manipulation!!)

Make sure to transform your variables as needed!

- CRIM: 0.15
- INDUS: 6.07
- CHAS: 1        
- RM:  6.1
- AGE: 33.2
- DIS: 7.6
- PTRATIO: 17
- B: 383
- LSTAT: 10.87
- RAD: 8
- TAX: 284

## Summary
Congratulations! You pre-processed the Boston Housing data using scaling and standardization. You also fitted your first multiple linear regression model on the Boston Housing data using statsmodels and scikit-learn!
