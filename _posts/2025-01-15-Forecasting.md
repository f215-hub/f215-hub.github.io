---
title: "Notes of ARIMA Time-Series Forecasting (updating)"
date: 2025-01-15T15:34:30-04:00
categories:
  - Blog
tags:
  - coding
  - python
---



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
```


```python
df = pd.read_csv('/Users/feiyu/graduate/Forecasting Project/superstore_sales.csv')
df.head()
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
      <th>Row ID</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Ship Date</th>
      <th>Ship Mode</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Segment</th>
      <th>Country</th>
      <th>City</th>
      <th>State</th>
      <th>Postal Code</th>
      <th>Region</th>
      <th>Product ID</th>
      <th>Category</th>
      <th>Sub-Category</th>
      <th>Product Name</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CA-2017-152156</td>
      <td>08/11/2017</td>
      <td>11/11/2017</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>Kentucky</td>
      <td>42420.0</td>
      <td>South</td>
      <td>FUR-BO-10001798</td>
      <td>Furniture</td>
      <td>Bookcases</td>
      <td>Bush Somerset Collection Bookcase</td>
      <td>261.9600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>CA-2017-152156</td>
      <td>08/11/2017</td>
      <td>11/11/2017</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>Kentucky</td>
      <td>42420.0</td>
      <td>South</td>
      <td>FUR-CH-10000454</td>
      <td>Furniture</td>
      <td>Chairs</td>
      <td>Hon Deluxe Fabric Upholstered Stacking Chairs,...</td>
      <td>731.9400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CA-2017-138688</td>
      <td>12/06/2017</td>
      <td>16/06/2017</td>
      <td>Second Class</td>
      <td>DV-13045</td>
      <td>Darrin Van Huff</td>
      <td>Corporate</td>
      <td>United States</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>90036.0</td>
      <td>West</td>
      <td>OFF-LA-10000240</td>
      <td>Office Supplies</td>
      <td>Labels</td>
      <td>Self-Adhesive Address Labels for Typewriters b...</td>
      <td>14.6200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>US-2016-108966</td>
      <td>11/10/2016</td>
      <td>18/10/2016</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>Florida</td>
      <td>33311.0</td>
      <td>South</td>
      <td>FUR-TA-10000577</td>
      <td>Furniture</td>
      <td>Tables</td>
      <td>Bretford CR4500 Series Slim Rectangular Table</td>
      <td>957.5775</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>US-2016-108966</td>
      <td>11/10/2016</td>
      <td>18/10/2016</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>Florida</td>
      <td>33311.0</td>
      <td>South</td>
      <td>OFF-ST-10000760</td>
      <td>Office Supplies</td>
      <td>Storage</td>
      <td>Eldon Fold 'N Roll Cart System</td>
      <td>22.3680</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9800 entries, 0 to 9799
    Data columns (total 18 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Row ID         9800 non-null   int64  
     1   Order ID       9800 non-null   object 
     2   Order Date     9800 non-null   object 
     3   Ship Date      9800 non-null   object 
     4   Ship Mode      9800 non-null   object 
     5   Customer ID    9800 non-null   object 
     6   Customer Name  9800 non-null   object 
     7   Segment        9800 non-null   object 
     8   Country        9800 non-null   object 
     9   City           9800 non-null   object 
     10  State          9800 non-null   object 
     11  Postal Code    9789 non-null   float64
     12  Region         9800 non-null   object 
     13  Product ID     9800 non-null   object 
     14  Category       9800 non-null   object 
     15  Sub-Category   9800 non-null   object 
     16  Product Name   9800 non-null   object 
     17  Sales          9800 non-null   float64
    dtypes: float64(2), int64(1), object(15)
    memory usage: 1.3+ MB



```python
df['Order Date'] = pd.to_datetime(df['Order Date'],format='%d/%m/%Y')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9800 entries, 0 to 9799
    Data columns (total 18 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   Row ID         9800 non-null   int64         
     1   Order ID       9800 non-null   object        
     2   Order Date     9800 non-null   datetime64[ns]
     3   Ship Date      9800 non-null   object        
     4   Ship Mode      9800 non-null   object        
     5   Customer ID    9800 non-null   object        
     6   Customer Name  9800 non-null   object        
     7   Segment        9800 non-null   object        
     8   Country        9800 non-null   object        
     9   City           9800 non-null   object        
     10  State          9800 non-null   object        
     11  Postal Code    9789 non-null   float64       
     12  Region         9800 non-null   object        
     13  Product ID     9800 non-null   object        
     14  Category       9800 non-null   object        
     15  Sub-Category   9800 non-null   object        
     16  Product Name   9800 non-null   object        
     17  Sales          9800 non-null   float64       
    dtypes: datetime64[ns](1), float64(2), int64(1), object(14)
    memory usage: 1.3+ MB



```python
# regroup sales by month
monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum()

# another method
# 
```


```python
monthly_sales.head()
```




    Order Date
    2015-01-31    14205.707
    2015-02-28     4519.892
    2015-03-31    55205.797
    2015-04-30    27906.855
    2015-05-31    23644.303
    Freq: M, Name: Sales, dtype: float64




```python
plt.figure(figsize=(15, 6))
plt.plot(monthly_sales, label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
```


    
![png](Forecasting_files/Forecasting_7_0.png)
    



```python
# y(t) = Level + Trend + Seasonality + Residual
decomposition = seasonal_decompose(monthly_sales, model='additive')
decomposition.plot().set_size_inches(14,10)
plt.show()
```


    
![png](Forecasting_files/Forecasting_8_0.png)
    



```python
# stationary test

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Augmented Dickey-Fuller Test Results:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return result[1] < 0.05

```


```python
is_stationary = check_stationarity(monthly_sales)
print(f"\nTime series is {'stationary' if is_stationary else 'non-stationary'}")
```

    Augmented Dickey-Fuller Test Results:
    ADF Statistic: -4.416136761430768
    p-value: 0.00027791039276670677
    Critical values:
    	1%: -3.5778480370438146
    	5%: -2.925338105429433
    	10%: -2.6007735310095064
    
    Time series is stationary



```python
# or test risidual to see stationarity 
residuals = decomposition.resid.dropna()  # avoid NaN
is_stationary = check_stationarity(residuals)
print(f"\nTime series is {'stationary' if is_stationary else 'non-stationary'}")
```

    Augmented Dickey-Fuller Test Results:
    ADF Statistic: -4.885259028495198
    p-value: 3.725318702838154e-05
    Critical values:
    	1%: -3.6699197407407405
    	5%: -2.9640707407407407
    	10%: -2.621171111111111
    
    Time series is stationary



```python

```
