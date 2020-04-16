
# Multiple Linear Regression in Statsmodels - Lab

## Introduction
In this lab, you'll practice fitting a multiple linear regression model on the Ames Housing dataset!

## Objectives
You will be able to:
* Determine if it is necessary to perform normalization/standardization for a specific model or set of data
* Use standardization/normalization on features of a dataset
* Identify if it is necessary to perform log transformations on a set of features
* Perform log transformations on different features of a dataset
* Use statsmodels to fit a multiple linear regression model
* Evaluate a linear regression model by using statistical performance metrics pertaining to overall model and specific parameters


## The Ames Housing Data

Using the specified continuous and categorical features, preprocess your data to prepare for modeling:
* Split off and one hot encode the categorical features of interest
* Log and scale the selected continuous features


```python
import pandas as pd
import numpy as np

ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']

```


```python
# __SOLUTION__

import pandas as pd
import numpy as np

ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']
```

## Continuous Features


```python
# Log transform and normalize
```


```python
# __SOLUTION__
# Log transform and normalize
ames_cont = ames[continuous]

# log features
log_names = [f'{column}_log' for column in ames_cont.columns]

ames_log = np.log(ames_cont)
ames_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

ames_log_norm = ames_log.apply(normalize)
```

## Categorical Features


```python
# One hot encode categoricals
```


```python
# __SOLUTION__
ames_ohe = pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)
```

## Combine Categorical and Continuous Features


```python
# combine features into a single dataframe called preprocessed
```


```python
# __SOLUTION__
preprocessed = pd.concat([ames_log_norm, ames_ohe], axis=1)
preprocessed.head()
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
      <th>LotArea_log</th>
      <th>1stFlrSF_log</th>
      <th>GrLivArea_log</th>
      <th>SalePrice_log</th>
      <th>BldgType_2fmCon</th>
      <th>BldgType_Duplex</th>
      <th>BldgType_Twnhs</th>
      <th>BldgType_TwnhsE</th>
      <th>KitchenQual_Fa</th>
      <th>KitchenQual_Gd</th>
      <th>...</th>
      <th>Neighborhood_NoRidge</th>
      <th>Neighborhood_NridgHt</th>
      <th>Neighborhood_OldTown</th>
      <th>Neighborhood_SWISU</th>
      <th>Neighborhood_Sawyer</th>
      <th>Neighborhood_SawyerW</th>
      <th>Neighborhood_Somerst</th>
      <th>Neighborhood_StoneBr</th>
      <th>Neighborhood_Timber</th>
      <th>Neighborhood_Veenker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.133185</td>
      <td>-0.803295</td>
      <td>0.529078</td>
      <td>0.559876</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.113403</td>
      <td>0.418442</td>
      <td>-0.381715</td>
      <td>0.212692</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.419917</td>
      <td>-0.576363</td>
      <td>0.659449</td>
      <td>0.733795</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.103311</td>
      <td>-0.439137</td>
      <td>0.541326</td>
      <td>-0.437232</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.878108</td>
      <td>0.112229</td>
      <td>1.281751</td>
      <td>1.014303</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



## Run a linear model with SalePrice as the target variable in statsmodels


```python
# Your code here
```


```python
# __SOLUTION__
X = preprocessed.drop('SalePrice_log', axis=1)
y = preprocessed['SalePrice_log']
```


```python
# __SOLUTION__ 
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```

    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>SalePrice_log</td>  <th>  R-squared:         </th> <td>   0.839</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.834</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   156.5</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 16 Apr 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>13:06:38</td>     <th>  Log-Likelihood:    </th> <td> -738.14</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1460</td>      <th>  AIC:               </th> <td>   1572.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1412</td>      <th>  BIC:               </th> <td>   1826.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    47</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                <td>   -0.1317</td> <td>    0.263</td> <td>   -0.500</td> <td> 0.617</td> <td>   -0.648</td> <td>    0.385</td>
</tr>
<tr>
  <th>LotArea_log</th>          <td>    0.1033</td> <td>    0.019</td> <td>    5.475</td> <td> 0.000</td> <td>    0.066</td> <td>    0.140</td>
</tr>
<tr>
  <th>1stFlrSF_log</th>         <td>    0.1371</td> <td>    0.016</td> <td>    8.584</td> <td> 0.000</td> <td>    0.106</td> <td>    0.168</td>
</tr>
<tr>
  <th>GrLivArea_log</th>        <td>    0.3768</td> <td>    0.016</td> <td>   24.114</td> <td> 0.000</td> <td>    0.346</td> <td>    0.407</td>
</tr>
<tr>
  <th>BldgType_2fmCon</th>      <td>   -0.1715</td> <td>    0.079</td> <td>   -2.173</td> <td> 0.030</td> <td>   -0.326</td> <td>   -0.017</td>
</tr>
<tr>
  <th>BldgType_Duplex</th>      <td>   -0.4203</td> <td>    0.062</td> <td>   -6.813</td> <td> 0.000</td> <td>   -0.541</td> <td>   -0.299</td>
</tr>
<tr>
  <th>BldgType_Twnhs</th>       <td>   -0.1403</td> <td>    0.093</td> <td>   -1.513</td> <td> 0.130</td> <td>   -0.322</td> <td>    0.042</td>
</tr>
<tr>
  <th>BldgType_TwnhsE</th>      <td>   -0.0512</td> <td>    0.060</td> <td>   -0.858</td> <td> 0.391</td> <td>   -0.168</td> <td>    0.066</td>
</tr>
<tr>
  <th>KitchenQual_Fa</th>       <td>   -0.9999</td> <td>    0.088</td> <td>  -11.315</td> <td> 0.000</td> <td>   -1.173</td> <td>   -0.827</td>
</tr>
<tr>
  <th>KitchenQual_Gd</th>       <td>   -0.3820</td> <td>    0.050</td> <td>   -7.613</td> <td> 0.000</td> <td>   -0.480</td> <td>   -0.284</td>
</tr>
<tr>
  <th>KitchenQual_TA</th>       <td>   -0.6692</td> <td>    0.055</td> <td>  -12.111</td> <td> 0.000</td> <td>   -0.778</td> <td>   -0.561</td>
</tr>
<tr>
  <th>SaleType_CWD</th>         <td>    0.2285</td> <td>    0.215</td> <td>    1.061</td> <td> 0.289</td> <td>   -0.194</td> <td>    0.651</td>
</tr>
<tr>
  <th>SaleType_Con</th>         <td>    0.5861</td> <td>    0.304</td> <td>    1.927</td> <td> 0.054</td> <td>   -0.010</td> <td>    1.183</td>
</tr>
<tr>
  <th>SaleType_ConLD</th>       <td>    0.3151</td> <td>    0.155</td> <td>    2.029</td> <td> 0.043</td> <td>    0.010</td> <td>    0.620</td>
</tr>
<tr>
  <th>SaleType_ConLI</th>       <td>    0.0331</td> <td>    0.195</td> <td>    0.169</td> <td> 0.865</td> <td>   -0.350</td> <td>    0.416</td>
</tr>
<tr>
  <th>SaleType_ConLw</th>       <td>    0.0161</td> <td>    0.196</td> <td>    0.082</td> <td> 0.935</td> <td>   -0.368</td> <td>    0.400</td>
</tr>
<tr>
  <th>SaleType_New</th>         <td>    0.2999</td> <td>    0.079</td> <td>    3.803</td> <td> 0.000</td> <td>    0.145</td> <td>    0.455</td>
</tr>
<tr>
  <th>SaleType_Oth</th>         <td>    0.1178</td> <td>    0.245</td> <td>    0.480</td> <td> 0.631</td> <td>   -0.364</td> <td>    0.599</td>
</tr>
<tr>
  <th>SaleType_WD</th>          <td>    0.1748</td> <td>    0.065</td> <td>    2.676</td> <td> 0.008</td> <td>    0.047</td> <td>    0.303</td>
</tr>
<tr>
  <th>MSZoning_FV</th>          <td>    1.0666</td> <td>    0.193</td> <td>    5.526</td> <td> 0.000</td> <td>    0.688</td> <td>    1.445</td>
</tr>
<tr>
  <th>MSZoning_RH</th>          <td>    0.8768</td> <td>    0.194</td> <td>    4.512</td> <td> 0.000</td> <td>    0.496</td> <td>    1.258</td>
</tr>
<tr>
  <th>MSZoning_RL</th>          <td>    0.9961</td> <td>    0.162</td> <td>    6.151</td> <td> 0.000</td> <td>    0.678</td> <td>    1.314</td>
</tr>
<tr>
  <th>MSZoning_RM</th>          <td>    1.1023</td> <td>    0.152</td> <td>    7.264</td> <td> 0.000</td> <td>    0.805</td> <td>    1.400</td>
</tr>
<tr>
  <th>Street_Pave</th>          <td>   -0.2131</td> <td>    0.180</td> <td>   -1.182</td> <td> 0.237</td> <td>   -0.567</td> <td>    0.141</td>
</tr>
<tr>
  <th>Neighborhood_Blueste</th> <td>    0.0529</td> <td>    0.318</td> <td>    0.167</td> <td> 0.868</td> <td>   -0.571</td> <td>    0.677</td>
</tr>
<tr>
  <th>Neighborhood_BrDale</th>  <td>   -0.4627</td> <td>    0.171</td> <td>   -2.711</td> <td> 0.007</td> <td>   -0.798</td> <td>   -0.128</td>
</tr>
<tr>
  <th>Neighborhood_BrkSide</th> <td>   -0.6498</td> <td>    0.137</td> <td>   -4.735</td> <td> 0.000</td> <td>   -0.919</td> <td>   -0.381</td>
</tr>
<tr>
  <th>Neighborhood_ClearCr</th> <td>   -0.2102</td> <td>    0.144</td> <td>   -1.456</td> <td> 0.146</td> <td>   -0.493</td> <td>    0.073</td>
</tr>
<tr>
  <th>Neighborhood_CollgCr</th> <td>   -0.0761</td> <td>    0.119</td> <td>   -0.641</td> <td> 0.522</td> <td>   -0.309</td> <td>    0.157</td>
</tr>
<tr>
  <th>Neighborhood_Crawfor</th> <td>   -0.0823</td> <td>    0.129</td> <td>   -0.638</td> <td> 0.523</td> <td>   -0.335</td> <td>    0.171</td>
</tr>
<tr>
  <th>Neighborhood_Edwards</th> <td>   -0.7613</td> <td>    0.124</td> <td>   -6.143</td> <td> 0.000</td> <td>   -1.004</td> <td>   -0.518</td>
</tr>
<tr>
  <th>Neighborhood_Gilbert</th> <td>   -0.0980</td> <td>    0.126</td> <td>   -0.777</td> <td> 0.437</td> <td>   -0.346</td> <td>    0.150</td>
</tr>
<tr>
  <th>Neighborhood_IDOTRR</th>  <td>   -0.9618</td> <td>    0.160</td> <td>   -6.014</td> <td> 0.000</td> <td>   -1.276</td> <td>   -0.648</td>
</tr>
<tr>
  <th>Neighborhood_MeadowV</th> <td>   -0.6918</td> <td>    0.159</td> <td>   -4.351</td> <td> 0.000</td> <td>   -1.004</td> <td>   -0.380</td>
</tr>
<tr>
  <th>Neighborhood_Mitchel</th> <td>   -0.2553</td> <td>    0.131</td> <td>   -1.944</td> <td> 0.052</td> <td>   -0.513</td> <td>    0.002</td>
</tr>
<tr>
  <th>Neighborhood_NAmes</th>   <td>   -0.4407</td> <td>    0.120</td> <td>   -3.664</td> <td> 0.000</td> <td>   -0.677</td> <td>   -0.205</td>
</tr>
<tr>
  <th>Neighborhood_NPkVill</th> <td>   -0.0160</td> <td>    0.173</td> <td>   -0.092</td> <td> 0.927</td> <td>   -0.356</td> <td>    0.324</td>
</tr>
<tr>
  <th>Neighborhood_NWAmes</th>  <td>   -0.2676</td> <td>    0.126</td> <td>   -2.122</td> <td> 0.034</td> <td>   -0.515</td> <td>   -0.020</td>
</tr>
<tr>
  <th>Neighborhood_NoRidge</th> <td>    0.3631</td> <td>    0.133</td> <td>    2.737</td> <td> 0.006</td> <td>    0.103</td> <td>    0.623</td>
</tr>
<tr>
  <th>Neighborhood_NridgHt</th> <td>    0.3626</td> <td>    0.120</td> <td>    3.029</td> <td> 0.002</td> <td>    0.128</td> <td>    0.597</td>
</tr>
<tr>
  <th>Neighborhood_OldTown</th> <td>   -0.9350</td> <td>    0.140</td> <td>   -6.686</td> <td> 0.000</td> <td>   -1.209</td> <td>   -0.661</td>
</tr>
<tr>
  <th>Neighborhood_SWISU</th>   <td>   -0.6998</td> <td>    0.144</td> <td>   -4.845</td> <td> 0.000</td> <td>   -0.983</td> <td>   -0.416</td>
</tr>
<tr>
  <th>Neighborhood_Sawyer</th>  <td>   -0.4754</td> <td>    0.128</td> <td>   -3.727</td> <td> 0.000</td> <td>   -0.726</td> <td>   -0.225</td>
</tr>
<tr>
  <th>Neighborhood_SawyerW</th> <td>   -0.2331</td> <td>    0.125</td> <td>   -1.860</td> <td> 0.063</td> <td>   -0.479</td> <td>    0.013</td>
</tr>
<tr>
  <th>Neighborhood_Somerst</th> <td>    0.0950</td> <td>    0.144</td> <td>    0.658</td> <td> 0.511</td> <td>   -0.188</td> <td>    0.378</td>
</tr>
<tr>
  <th>Neighborhood_StoneBr</th> <td>    0.4296</td> <td>    0.133</td> <td>    3.232</td> <td> 0.001</td> <td>    0.169</td> <td>    0.690</td>
</tr>
<tr>
  <th>Neighborhood_Timber</th>  <td>    0.0057</td> <td>    0.134</td> <td>    0.042</td> <td> 0.966</td> <td>   -0.257</td> <td>    0.269</td>
</tr>
<tr>
  <th>Neighborhood_Veenker</th> <td>    0.1276</td> <td>    0.169</td> <td>    0.754</td> <td> 0.451</td> <td>   -0.204</td> <td>    0.460</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>289.988</td> <th>  Durbin-Watson:     </th> <td>   1.967</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1242.992</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-0.886</td>  <th>  Prob(JB):          </th> <td>1.22e-270</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.159</td>  <th>  Cond. No.          </th> <td>    109.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Run the same model in scikit-learn


```python
# Your code here - Check that the coefficients and intercept are the same as those from Statsmodels
```


```python
# __SOLUTION__ 
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# __SOLUTION__ 
# coefficients
linreg.coef_
```




    array([ 0.10327192,  0.1371289 ,  0.37682133, -0.1714623 , -0.42033885,
           -0.14034113, -0.05120194, -0.99986001, -0.38202198, -0.66924909,
            0.22847737,  0.5860786 ,  0.31510567,  0.0330941 ,  0.01608664,
            0.29985338,  0.11784232,  0.17480326,  1.06663561,  0.87681007,
            0.99609131,  1.10228499, -0.21311107,  0.05293276, -0.46271253,
           -0.64982261, -0.21019239, -0.07609253, -0.08233633, -0.76126683,
           -0.09799942, -0.96183328, -0.69182575, -0.2553217 , -0.44067351,
           -0.01595046, -0.26762962,  0.36313165,  0.36259667, -0.93504972,
           -0.69976325, -0.47543141, -0.23309732,  0.09502969,  0.42957077,
            0.0056924 ,  0.12762613])




```python
# __SOLUTION__ 
# intercept
linreg.intercept_
```




    -0.13169736916667654



## Predict the house price given the following characteristics (before manipulation!!)

Make sure to transform your variables as needed!

- LotArea: 14977
- 1stFlrSF: 1976
- GrLivArea: 1976
- BldgType: 1Fam
- KitchenQual: Gd
- SaleType: New
- MSZoning: RL
- Street: Pave
- Neighborhood: NridgHt

## Summary
Congratulations! You pre-processed the Ames Housing data using scaling and standardization. You also fitted your first multiple linear regression model on the Ames Housing data using statsmodels and scikit-learn!
