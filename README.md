
# Multiple Linear Regression in Statsmodels - Lab

## Introduction
In this lab, you'll practice fitting a multiple linear regression model on our Boston Housing Data set!

## Objectives
You will be able to:
* Run linear regression on Boston Housing dataset with all the predictors
* Interpret the parameters of the multiple linear regression model

## The Boston Housing Data

We pre-processed the Boston Housing Data again. This time, however, we did things slightly different:
- We dropped "ZN" and "NOX" completely
- We categorized "RAD" in 3 bins and "TAX" in 4 bins
- We used min-max-scaling on "B", "CRIM" and "DIS" (and logtransformed all of them first, except "B")
- We used standardization on "AGE", "INDUS", "LSTAT" and "PTRATIO" (and logtransformed all of them first, except for "AGE") 


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_features = boston_features.drop(["NOX","ZN"],axis=1)

# first, create bins for based on the values observed. 3 values will result in 2 bins
bins = [0,6,  24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# first, create bins for based on the values observed. 4 values will result in 3 bins
bins = [0, 270, 360, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX")
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD")
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)
```


```python
age = boston_features["AGE"]
b = boston_features["B"]
logcrim = np.log(boston_features["CRIM"])
logdis = np.log(boston_features["DIS"])
logindus = np.log(boston_features["INDUS"])
loglstat = np.log(boston_features["LSTAT"])
logptratio = np.log(boston_features["PTRATIO"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["CRIM"] = (logcrim-min(logcrim))/(max(logcrim)-min(logcrim))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["AGE"] = (age-np.mean(age))/np.sqrt(np.var(age))
boston_features["INDUS"] = (logindus-np.mean(logindus))/np.sqrt(np.var(logindus))
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
boston_features["PTRATIO"] = (logptratio-np.mean(logptratio))/(np.sqrt(np.var(logptratio)))
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
      <th>RAD_(0, 6]</th>
      <th>RAD_(6, 24]</th>
      <th>TAX_(0, 270]</th>
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
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# __SOLUTION__ 
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_features = boston_features.drop(["NOX","ZN"],axis=1)

# first, create bins for based on the values observed. 3 values will result in 2 bins
bins = [0,6,  24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# first, create bins for based on the values observed. 4 values will result in 3 bins
bins = [0, 270, 360, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX")
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD")
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)
```


```python
# __SOLUTION__ 
age = boston_features["AGE"]
b = boston_features["B"]
logcrim = np.log(boston_features["CRIM"])
logdis = np.log(boston_features["DIS"])
logindus = np.log(boston_features["INDUS"])
loglstat = np.log(boston_features["LSTAT"])
logptratio = np.log(boston_features["PTRATIO"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["CRIM"] = (logcrim-min(logcrim))/(max(logcrim)-min(logcrim))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["AGE"] = (age-np.mean(age))/np.sqrt(np.var(age))
boston_features["INDUS"] = (logindus-np.mean(logindus))/np.sqrt(np.var(logindus))
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
boston_features["PTRATIO"] = (logptratio-np.mean(logptratio))/(np.sqrt(np.var(logptratio)))
```


```python
# __SOLUTION__ 
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
      <th>RAD_(0, 6]</th>
      <th>RAD_(6, 24]</th>
      <th>TAX_(0, 270]</th>
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
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Run an linear model in Statsmodels


```python

```


```python
# __SOLUTION__ 
X = boston_features
y = pd.DataFrame(boston.target, columns= ["price"])
```


```python
# __SOLUTION__ 
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
  <th>Date:</th>             <td>Sun, 14 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>5.15e-153</td>
</tr>
<tr>
  <th>Time:</th>                 <td>21:54:50</td>     <th>  Log-Likelihood:    </th> <td> -1458.2</td> 
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
  <th>const</th>          <td>    4.4397</td> <td>    1.785</td> <td>    2.487</td> <td> 0.013</td> <td>    0.933</td> <td>    7.947</td>
</tr>
<tr>
  <th>CRIM</th>           <td>   -1.9000</td> <td>    2.091</td> <td>   -0.909</td> <td> 0.364</td> <td>   -6.009</td> <td>    2.209</td>
</tr>
<tr>
  <th>INDUS</th>          <td>   -0.8069</td> <td>    0.362</td> <td>   -2.228</td> <td> 0.026</td> <td>   -1.518</td> <td>   -0.095</td>
</tr>
<tr>
  <th>CHAS</th>           <td>    2.5968</td> <td>    0.796</td> <td>    3.262</td> <td> 0.001</td> <td>    1.033</td> <td>    4.161</td>
</tr>
<tr>
  <th>RM</th>             <td>    2.6445</td> <td>    0.408</td> <td>    6.480</td> <td> 0.000</td> <td>    1.843</td> <td>    3.446</td>
</tr>
<tr>
  <th>AGE</th>            <td>    0.0787</td> <td>    0.352</td> <td>    0.224</td> <td> 0.823</td> <td>   -0.612</td> <td>    0.770</td>
</tr>
<tr>
  <th>DIS</th>            <td>  -10.0839</td> <td>    1.855</td> <td>   -5.437</td> <td> 0.000</td> <td>  -13.728</td> <td>   -6.440</td>
</tr>
<tr>
  <th>PTRATIO</th>        <td>   -1.4864</td> <td>    0.241</td> <td>   -6.159</td> <td> 0.000</td> <td>   -1.961</td> <td>   -1.012</td>
</tr>
<tr>
  <th>B</th>              <td>    3.8623</td> <td>    0.981</td> <td>    3.935</td> <td> 0.000</td> <td>    1.934</td> <td>    5.791</td>
</tr>
<tr>
  <th>LSTAT</th>          <td>   -5.6315</td> <td>    0.354</td> <td>  -15.929</td> <td> 0.000</td> <td>   -6.326</td> <td>   -4.937</td>
</tr>
<tr>
  <th>RAD_(0, 6]</th>     <td>    1.5563</td> <td>    0.821</td> <td>    1.896</td> <td> 0.059</td> <td>   -0.056</td> <td>    3.169</td>
</tr>
<tr>
  <th>RAD_(6, 24]</th>    <td>    2.8834</td> <td>    1.069</td> <td>    2.697</td> <td> 0.007</td> <td>    0.783</td> <td>    4.984</td>
</tr>
<tr>
  <th>TAX_(0, 270]</th>   <td>    2.6166</td> <td>    0.715</td> <td>    3.661</td> <td> 0.000</td> <td>    1.212</td> <td>    4.021</td>
</tr>
<tr>
  <th>TAX_(270, 360]</th> <td>    1.3553</td> <td>    0.702</td> <td>    1.930</td> <td> 0.054</td> <td>   -0.025</td> <td>    2.735</td>
</tr>
<tr>
  <th>TAX_(360, 712]</th> <td>    0.4679</td> <td>    0.683</td> <td>    0.685</td> <td> 0.493</td> <td>   -0.873</td> <td>    1.809</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>106.736</td> <th>  Durbin-Watson:     </th> <td>   1.093</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 431.931</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.891</td>  <th>  Prob(JB):          </th> <td>1.61e-94</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.161</td>  <th>  Cond. No.          </th> <td>5.68e+16</td>
</tr>
</table>



## Run the same model in Scikit-learn


```python

```


```python
# __SOLUTION__ 
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# __SOLUTION__ 
# coefficients
linreg.coef_
```




    array([[ -1.89995316,  -0.80688617,   2.59684028,   2.64453176,
              0.0786663 , -10.0839112 ,  -1.48638161,   3.86233002,
             -5.63145746,  -0.66357383,   0.66357383,   1.13664819,
             -0.12462051,  -1.01202768]])




```python
# __SOLUTION__ 
# intercept
linreg.intercept_
```




    array([8.13952371])



## Remove the necessary variables to make sure the coefficients are the same for Scikit-learn vs Statsmodels


```python

```


```python
# __SOLUTION__ 
X_smaller = X.drop(["RAD_(0, 6]", "TAX_(270, 360]"], axis=1)
X_smaller.head()
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
      <th>TAX_(0, 270]</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Statsmodels


```python

```


```python
# __SOLUTION__ 
import statsmodels.api as sm
X_int_sm = sm.add_constant(X_smaller)
model = sm.OLS(y,X_int_sm).fit()
# model.summary()
```


```python
# __SOLUTION__ 
model.params
```




    const              7.351329
    CRIM              -1.899953
    INDUS             -0.806886
    CHAS               2.596840
    RM                 2.644532
    AGE                0.078666
    DIS              -10.083911
    PTRATIO           -1.486382
    B                  3.862330
    LSTAT             -5.631457
    RAD_(6, 24]        1.327148
    TAX_(0, 270]       1.261269
    TAX_(360, 712]    -0.887407
    dtype: float64



### Scikit-learn


```python

```


```python
# __SOLUTION__ 
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_smaller, y)
print(linreg.intercept_)
print(linreg.coef_)
```

    [7.35132937]
    [[ -1.89995316  -0.80688617   2.59684028   2.64453176   0.0786663
      -10.0839112   -1.48638161   3.86233002  -5.63145746   1.32714767
        1.2612687   -0.88740717]]


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
Congratulations! You've fitted your first multiple linear regression model on the Boston Housing Data.
