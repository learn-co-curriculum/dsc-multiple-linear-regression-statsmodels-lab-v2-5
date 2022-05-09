# Multiple Linear Regression in StatsModels - Lab

## Introduction
In this lab, you'll practice fitting a multiple linear regression model on the Ames Housing dataset!

## Objectives

You will be able to:

* Perform a multiple linear regression using StatsModels
* Visualize individual predictors within a multiple linear regression
* Interpret multiple linear regression coefficients from raw, un-transformed data

## The Ames Housing Dataset

The [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf) is a newer (2011) replacement for the classic Boston Housing dataset. Each record represents a residential property sale in Ames, Iowa. It contains many different potential predictors and the target variable is `SalePrice`.


```python
import pandas as pd
ames = pd.read_csv("ames.csv", index_col=0)
ames
```


```python
ames.describe()
```

We will focus specifically on a subset of the overall dataset. These features are:

```
LotArea: Lot size in square feet

1stFlrSF: First Floor square feet

GrLivArea: Above grade (ground) living area square feet
```


```python
ames_subset = ames[['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']].copy()
ames_subset
```

## Step 1: Visualize Relationships Between Features and Target

For each feature in the subset, create a scatter plot that shows the feature on the x-axis and `SalePrice` on the y-axis.


```python
# Your code here - import relevant library, create scatter plots
```


```python
# Your written answer here - do these seem like good candidates for linear regression?
```

## Step 2: Build a Simple Linear Regression Model

Set the dependent variable (`y`) to be the `SalePrice`, then choose one of the features shown in the subset above to be the baseline independent variable (`X`).

Build a linear regression using StatsModels, describe the overall model performance, and interpret its coefficients.


```python
# Your code here - define y and baseline X
```


```python
# Your code here - import StatsModels, fit baseline model, display results
```


```python
# Your written answer here - interpret model results
```

## Step 3: Build a Multiple Linear Regression Model

For this model, use **all of** the features in `ames_subset`.


```python
# Your code here - define X
```


```python
# Your code here - fit model and display results
```


```python
# Your written answer here - interpret model results. Does this model seem better than the previous one?
```

## Step 4: Create Partial Regression Plots for Features

Using your model from Step 3, visualize each of the features using partial regression plots.


```python
# Your code here - create partial regression plots for each predictor
```


```python
# Your written answer here - explain what you see, and how this relates
# to what you saw in Step 1. What do you notice?
```

## Level Up (Optional)

Re-create this model in scikit-learn, and check if you get the same R-Squared and coefficients.


```python
# Your code here - import linear regression from scikit-learn and create and fit model
```


```python
# Your code here - compare R-Squared
```


```python
# Your code here - compare intercept and coefficients
```

## Summary
Congratulations! You fitted your first multiple linear regression model on the Ames Housing data using StatsModels.
