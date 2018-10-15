
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



## Run an linear model in Statsmodels


```python
X = boston_features
y = pd.DataFrame(boston.target, columns= ["price"])
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
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# coefficients
linreg.coef_
```




    array([[ -1.89995316,  -0.80688617,   2.59684028,   2.64453176,
              0.0786663 , -10.0839112 ,  -1.48638161,   3.86233002,
             -5.63145746,  -0.66357383,   0.66357383,   1.13664819,
             -0.12462051,  -1.01202768]])




```python
# intercept
linreg.intercept_
```




    array([8.13952371])



## Remove the necessary variables to make sure the coefficients are the same for Scikit-learn vs Statsmodels


```python
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
import statsmodels.api as sm
X_int_sm = sm.add_constant(X_smaller)
model = sm.OLS(y,X_int_sm).fit()
# model.summary()
```


```python
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


```python
bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos[200:250]
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
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>0.01778</td>
      <td>95.0</td>
      <td>1.47</td>
      <td>0.0</td>
      <td>0.4030</td>
      <td>7.135</td>
      <td>13.9</td>
      <td>7.6534</td>
      <td>3.0</td>
      <td>402.0</td>
      <td>17.0</td>
      <td>384.30</td>
      <td>4.45</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0.03445</td>
      <td>82.5</td>
      <td>2.03</td>
      <td>0.0</td>
      <td>0.4150</td>
      <td>6.162</td>
      <td>38.4</td>
      <td>6.2700</td>
      <td>2.0</td>
      <td>348.0</td>
      <td>14.7</td>
      <td>393.77</td>
      <td>7.43</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.02177</td>
      <td>82.5</td>
      <td>2.03</td>
      <td>0.0</td>
      <td>0.4150</td>
      <td>7.610</td>
      <td>15.7</td>
      <td>6.2700</td>
      <td>2.0</td>
      <td>348.0</td>
      <td>14.7</td>
      <td>395.38</td>
      <td>3.11</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.03510</td>
      <td>95.0</td>
      <td>2.68</td>
      <td>0.0</td>
      <td>0.4161</td>
      <td>7.853</td>
      <td>33.2</td>
      <td>5.1180</td>
      <td>4.0</td>
      <td>224.0</td>
      <td>14.7</td>
      <td>392.78</td>
      <td>3.81</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.02009</td>
      <td>95.0</td>
      <td>2.68</td>
      <td>0.0</td>
      <td>0.4161</td>
      <td>8.034</td>
      <td>31.9</td>
      <td>5.1180</td>
      <td>4.0</td>
      <td>224.0</td>
      <td>14.7</td>
      <td>390.55</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0.13642</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>5.891</td>
      <td>22.3</td>
      <td>3.9454</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>396.90</td>
      <td>10.87</td>
    </tr>
    <tr>
      <th>206</th>
      <td>0.22969</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>6.326</td>
      <td>52.5</td>
      <td>4.3549</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>394.87</td>
      <td>10.97</td>
    </tr>
    <tr>
      <th>207</th>
      <td>0.25199</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>5.783</td>
      <td>72.7</td>
      <td>4.3549</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>389.43</td>
      <td>18.06</td>
    </tr>
    <tr>
      <th>208</th>
      <td>0.13587</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>6.064</td>
      <td>59.1</td>
      <td>4.2392</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>381.32</td>
      <td>14.66</td>
    </tr>
    <tr>
      <th>209</th>
      <td>0.43571</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>5.344</td>
      <td>100.0</td>
      <td>3.8750</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>396.90</td>
      <td>23.09</td>
    </tr>
    <tr>
      <th>210</th>
      <td>0.17446</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>5.960</td>
      <td>92.1</td>
      <td>3.8771</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>393.25</td>
      <td>17.27</td>
    </tr>
    <tr>
      <th>211</th>
      <td>0.37578</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>5.404</td>
      <td>88.6</td>
      <td>3.6650</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>395.24</td>
      <td>23.98</td>
    </tr>
    <tr>
      <th>212</th>
      <td>0.21719</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>5.807</td>
      <td>53.8</td>
      <td>3.6526</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>390.94</td>
      <td>16.03</td>
    </tr>
    <tr>
      <th>213</th>
      <td>0.14052</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>6.375</td>
      <td>32.3</td>
      <td>3.9454</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>385.81</td>
      <td>9.38</td>
    </tr>
    <tr>
      <th>214</th>
      <td>0.28955</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>5.412</td>
      <td>9.8</td>
      <td>3.5875</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>348.93</td>
      <td>29.55</td>
    </tr>
    <tr>
      <th>215</th>
      <td>0.19802</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.4890</td>
      <td>6.182</td>
      <td>42.4</td>
      <td>3.9454</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>393.63</td>
      <td>9.47</td>
    </tr>
    <tr>
      <th>216</th>
      <td>0.04560</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>1.0</td>
      <td>0.5500</td>
      <td>5.888</td>
      <td>56.0</td>
      <td>3.1121</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>392.80</td>
      <td>13.51</td>
    </tr>
    <tr>
      <th>217</th>
      <td>0.07013</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>0.0</td>
      <td>0.5500</td>
      <td>6.642</td>
      <td>85.1</td>
      <td>3.4211</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>392.78</td>
      <td>9.69</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0.11069</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>1.0</td>
      <td>0.5500</td>
      <td>5.951</td>
      <td>93.8</td>
      <td>2.8893</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>396.90</td>
      <td>17.92</td>
    </tr>
    <tr>
      <th>219</th>
      <td>0.11425</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>1.0</td>
      <td>0.5500</td>
      <td>6.373</td>
      <td>92.4</td>
      <td>3.3633</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>393.74</td>
      <td>10.50</td>
    </tr>
    <tr>
      <th>220</th>
      <td>0.35809</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.5070</td>
      <td>6.951</td>
      <td>88.5</td>
      <td>2.8617</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>391.70</td>
      <td>9.71</td>
    </tr>
    <tr>
      <th>221</th>
      <td>0.40771</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.5070</td>
      <td>6.164</td>
      <td>91.3</td>
      <td>3.0480</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>395.24</td>
      <td>21.46</td>
    </tr>
    <tr>
      <th>222</th>
      <td>0.62356</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.5070</td>
      <td>6.879</td>
      <td>77.7</td>
      <td>3.2721</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>390.39</td>
      <td>9.93</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0.61470</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5070</td>
      <td>6.618</td>
      <td>80.8</td>
      <td>3.2721</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>396.90</td>
      <td>7.60</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.31533</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>8.266</td>
      <td>78.3</td>
      <td>2.8944</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>385.05</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>225</th>
      <td>0.52693</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>8.725</td>
      <td>83.0</td>
      <td>2.8944</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>382.00</td>
      <td>4.63</td>
    </tr>
    <tr>
      <th>226</th>
      <td>0.38214</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>8.040</td>
      <td>86.5</td>
      <td>3.2157</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>387.38</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>227</th>
      <td>0.41238</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>7.163</td>
      <td>79.9</td>
      <td>3.2157</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>372.08</td>
      <td>6.36</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0.29819</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>7.686</td>
      <td>17.0</td>
      <td>3.3751</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>377.51</td>
      <td>3.92</td>
    </tr>
    <tr>
      <th>229</th>
      <td>0.44178</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>6.552</td>
      <td>21.4</td>
      <td>3.3751</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>380.34</td>
      <td>3.76</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0.53700</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>5.981</td>
      <td>68.1</td>
      <td>3.6715</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>378.35</td>
      <td>11.65</td>
    </tr>
    <tr>
      <th>231</th>
      <td>0.46296</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>7.412</td>
      <td>76.9</td>
      <td>3.6715</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>376.14</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>232</th>
      <td>0.57529</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5070</td>
      <td>8.337</td>
      <td>73.3</td>
      <td>3.8384</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>385.91</td>
      <td>2.47</td>
    </tr>
    <tr>
      <th>233</th>
      <td>0.33147</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5070</td>
      <td>8.247</td>
      <td>70.4</td>
      <td>3.6519</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>378.95</td>
      <td>3.95</td>
    </tr>
    <tr>
      <th>234</th>
      <td>0.44791</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.5070</td>
      <td>6.726</td>
      <td>66.5</td>
      <td>3.6519</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>360.20</td>
      <td>8.05</td>
    </tr>
    <tr>
      <th>235</th>
      <td>0.33045</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5070</td>
      <td>6.086</td>
      <td>61.5</td>
      <td>3.6519</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>376.75</td>
      <td>10.88</td>
    </tr>
    <tr>
      <th>236</th>
      <td>0.52058</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.5070</td>
      <td>6.631</td>
      <td>76.5</td>
      <td>4.1480</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>388.45</td>
      <td>9.54</td>
    </tr>
    <tr>
      <th>237</th>
      <td>0.51183</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5070</td>
      <td>7.358</td>
      <td>71.6</td>
      <td>4.1480</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>390.07</td>
      <td>4.73</td>
    </tr>
    <tr>
      <th>238</th>
      <td>0.08244</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.481</td>
      <td>18.5</td>
      <td>6.1899</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>379.41</td>
      <td>6.36</td>
    </tr>
    <tr>
      <th>239</th>
      <td>0.09252</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.606</td>
      <td>42.2</td>
      <td>6.1899</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>383.78</td>
      <td>7.37</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0.11329</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.897</td>
      <td>54.3</td>
      <td>6.3361</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>391.25</td>
      <td>11.38</td>
    </tr>
    <tr>
      <th>241</th>
      <td>0.10612</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.095</td>
      <td>65.1</td>
      <td>6.3361</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>394.62</td>
      <td>12.40</td>
    </tr>
    <tr>
      <th>242</th>
      <td>0.10290</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.358</td>
      <td>52.9</td>
      <td>7.0355</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>372.75</td>
      <td>11.22</td>
    </tr>
    <tr>
      <th>243</th>
      <td>0.12757</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.393</td>
      <td>7.8</td>
      <td>7.0355</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>374.71</td>
      <td>5.19</td>
    </tr>
    <tr>
      <th>244</th>
      <td>0.20608</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>5.593</td>
      <td>76.5</td>
      <td>7.9549</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>372.49</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>245</th>
      <td>0.19133</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>5.605</td>
      <td>70.2</td>
      <td>7.9549</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>389.13</td>
      <td>18.46</td>
    </tr>
    <tr>
      <th>246</th>
      <td>0.33983</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>6.108</td>
      <td>34.9</td>
      <td>8.0555</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>390.18</td>
      <td>9.16</td>
    </tr>
    <tr>
      <th>247</th>
      <td>0.19657</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>6.226</td>
      <td>79.2</td>
      <td>8.0555</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>376.14</td>
      <td>10.15</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.16439</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>6.433</td>
      <td>49.1</td>
      <td>7.8265</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>374.71</td>
      <td>9.52</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.19073</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>6.718</td>
      <td>17.5</td>
      <td>7.8265</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>393.74</td>
      <td>6.56</td>
    </tr>
  </tbody>
</table>
</div>



## Summary
Congratulations! You've fitted your first multiple linear regression model on the Boston Housing Data.
