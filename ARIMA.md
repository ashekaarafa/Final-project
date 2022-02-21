# Hydrological Time Series Analysis at Hofkirchen Danube River

![rhone](https://github.com/sahasandip/Final-project/blob/cd1097b89d92d9c5154296f89b1a2826fb43c737/danube.jpg)


***
>   ***Goals***: Analysis the abfluss data of Hofkirchen, Danube, Visual representation of water level variation in time, Next 10 days water level forecast with autoregressive integrated moving average(ARIMA) model.

>   ***Requirements***: *Python* libraries: *numpy* including *scipy* and *matplotlib*. Read and understand the [data handling with *numpy*](https://hydro-informatics.github.io/hypy_pynum.html) and [functions](https://hydro-informatics.github.io/hypy_pyfun.html).

Get ready by cloning the repository:

```
git clone https://github.com/Ecohydraulics/Exercise-SequentPeak.git
```
## Theory
An autoregressive integrated moving average(ARIMA) is a statistical analysis model that uses time series data 
to either better understand the data set or to predict future trends. This model predicts future values based on past values. 
It has three parameters (p,d,q).
Where 

p is the order of the AR term,
q is the order of the MA term,
d is the number of differencing required to make the time series stationary.


Arima model is applied with train data set, 
afterwards, It is compared with the actual test data through visualization. The error with chosen order of p,d,q is calculated for model evaluation. 

## Abfluss (Flow data) data
The daily water level at Hofkirchen, Danube River are available from 2021.11.19  through 2022.01.24 in the form of `.csv` files 
([`flows` folder](https://github.com/Ecohydraulics/Exercise-SequentPeak/tree/master/flows)). The data is downloded from the link below()
The name of the file is set according to the dates example: 01.01.2022.csv. Each file contains water level data (mÂ³/s) at 15 mins interval.
Missing values are written as XXX,XXX. The data has header.

### File list:

1. my_config.py : Import all necessary modules and packages
2. package folder "mypkg" contains three files
      __init__.py 
      dataedit.py : Contains class DataRead, for pre-processing  the data and daily average calculation.
      modelling.py : Contains class Modelling, for forecasting with ARIMA model through calculating least errors. 
      classical_arima.py: Contains class Classical. Implementing ARIMA model after manually selecting the parameters (p, d, q).
3. main.py: The above  files are plugged in "main.py" to calculate the future water levels.
4. main_2.py: create several plots (autocorrelation, partial autocorrelation plot). After visualizing the graph parameters of Arima model are selected.

5. myapp.py: It is an "Water level Forecast"app. 




 
###my_config.py: 
import necessary modules and packages.

import numpy as np
import pandas as pd
import itertools
import warnings
import os
import sys
import glob
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import mypkg as pkg

 ###main.py: 
 
from my_config import *

csv_files = glob.glob(os.path.join("my_data", "*.csv"))  

will create a list similar like this 

['my_data\\01.01.2022.csv', 'my_data\\01.12.2021.csv', 'my_data\\02.01.2022.csv', 'my_data\\02.12.2021.csv', ....
 
1.get_average_df(): seperate the filename from the "csv_files" . 
					pathname and file name is plugged in to Dataread class.
					data1 = DataRead("my_data", i[-1]) here i contains the file name.
					average value is calculated for each file.
					A data frame "myy_df" is created with "Date", "Average_Water_Level"
					The dataframe will rewrite or create a new file called "modified_average_file.csv".
					The file will look like this:
					
Date,Average_Water_Level
2021-11-19,356.09
2021-11-20,367.71
2021-11-23,337.34
2021-11-24,332.58
2021-11-25,323.83
2021-11-29,328.97
2021-11-30,343.31
2021-12-01,337.13
2021-12-02,337.13
2021-12-03,460.91
2021-12-04,487.91
2021-12-05,478.04
2021-12-06,605.44
2021-12-07,602.0
2021-12-08,536.69
2021-12-09,503.0
2021-12-10,499.72
2021-12-11,482.3
2021-12-12,467.37
2021-12-13,450.1
2021-12-14,461.39
2021-12-15,539.98
2021-12-16,577.33
2021-12-17,533.85
2021-12-18,520.15
2021-12-19,504.06
2021-12-20,484.63
2021-12-27,654.17
2021-12-28,627.62
2021-12-29,713.47
2021-12-30,965.32
2021-12-31,1250.31
2022-01-01,1296.56
2022-01-02,1080.64
2022-01-03,900.05
2022-01-04,846.71
2022-01-05,919.93
2022-01-06,1092.4
2022-01-07,1127.6
2022-01-08,968.71
2022-01-09,868.52
2022-01-10,774.46
2022-01-11,750.33
2022-01-12,709.67
2022-01-13,673.35
2022-01-14,645.24
2022-01-15,611.23
2022-01-16,574.36
2022-01-17,542.34
2022-01-18,535.57
2022-01-19,552.51
2022-01-20,545.34
2022-01-21,535.33
2022-01-22,535.06
2022-01-23,524.02
2022-01-24,523.04


2. my_forecast(): 
		daily average data frame is plugged in to class Modelling.
         frame = get_average_df()
         data_g = Modelling(frame)
		Afterwards, predicted data is calculated for the next 10 days.

		running the main.py file will print this:
		
the root mean square error of our model is 9.694525351159205
the correlation coefficient is 0.7835117705002723
        Date  Estimated_Water_Level
0 2022-01-24             528.947969
1 2022-01-25             527.827890
2 2022-01-26             527.097135
3 2022-01-27             521.790380
4 2022-01-28             519.915409
5 2022-01-29             516.412162
6 2022-01-30             515.784846
7 2022-01-31             513.064404
8 2022-02-01             511.926265
9 2022-02-02             508.939543



3.forecast_plots(): will plot a graph of daily average water level. You can download the graph figure from here link ()
	train = plotting of 50 days data (train set)
    actual = plotting of the rest days data (test set)
    my model = our model predicted with train data set upto the length of test set.
    my forecast = our model prediction for next 10 days with overall timeseries
	
	
###main_2.py
from my_config import *


Read the modified_average_file.csv 

my_df = pd.read_csv("modified_average_file.csv", header=0,
                    index_col=False,
                    skipinitialspace=True,
                    skip_blank_lines=True)
	  
1.get_d(): create auto-correlation plot for timeseries for differention order 0, 1, 2 . Also do stationary check. 
after observing the acf plot we select in the prompt d = 0; Though the data is not stationary we select d=0. 
Since for d=1, d=2  it looks over differencing. 
this function will return the value of d 
link of the fig: githublink
2.get_my_order(): create two plots pacf and acf with the timeseries. Degree of diffencing is set from the entered value of d.
Visulizing the graph pacf plot we observe that one data goes much lower. So we consider p = 1. 
From acf plot we select our q parameter = 0. Expert opinion is always good to select the parameters through graph visualization.
return a tuple (p,d,q)=(1,0,0)

3.get_forecast(): return the forecast values. 

running the main_2.py will show similar to this.


C:\Users\asheka\anaconda3\python.exe D:/pythonProject/danube/main_2.py
Results of Dickey-Fuller Test:
p-value: 0.413846
ADF statistics:-1.733684
p value greater than 0.05.Data is non stationary
0
Enter d:0
Enter p:1
Enter q:0
my chosen order (p, d, q) for ARIMA model is (1, 0, 0) 
model error(rmse) is 14.945323440795713 and correlation coefficient is -0.9798003549329838
-1
Next 10 days water level for Hofkirchen station
56    525.650542
57    528.078763
58    530.337398
59    532.438290
60    534.392455
61    536.210142
62    537.900881
63    539.473539
64    540.936363
65    542.297023
Name: predicted_mean, dtype: float64

Process finished with exit code 0


### myapp.py: 

first you need to import all necessary modules and libraries.

import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from dataedit import DataRead
from modelling import Modelling
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from my_config import *


class ForecastApp(tk.Frame): contain methods below

select_file(): open directory to select the daily average csv file.
get_forecasting(): Compute the next water level data using class Modelling.
get_info(): Show up the water level result for selected day.

necessary info for running myapp.py:
step 1: select "modified_average_file.csv" from the directory.
step 2: Click compute. 
step 3: Select the date from the combo list.
step 4: click "check" button to see the result.

package folder "mypkg" with an __init__ file contains three file with three class.
					
###dataedit.py file:
class DataRead contains methods. parameter are pathname, filename, delimeter =","
1.get_data_frame(): This method will read the csv file through path and file name.


if __name__ == '__main__':

    data1 = DataRead("insert path of my_data folder,", "01.12.2021.csv")
    print(data1.get_dataframe())

Look similar to this.

   01.12.2021;"WSA Regensburg";"DONAU";"HOFKIRCHEN";"10088003";"Q_P";"m³/s";"XXX  ...  6"
0                                         00:15;"340"                             ... NaN
1                                         00:30;"340"                             ... NaN
2                                         00:45;"343"                             ... NaN
3                                         01:00;"340"                             ... NaN
4                                         01:15;"340"                             ... NaN
..                                                ...                             ...  ..
91                                        23:00;"340"                             ... NaN
92                                        23:15;"343"                             ... NaN
93                                        23:30;"343"                             ... NaN
94                                         23:45;"XXX                             ... NaN
95                                        24:00;"343"                             ... NaN

[96 rows x 3 columns]

Process finished with exit code 0

2.get_time_list(): Seperate the time from the dataframe and return the time sequence as list. 24:00 time value is replaced with 
23:59 to avoid confliction. Because we face difficulties with further data processing, 24:00 is recognised as the next day.

3.get_level_list(): Seperate the water level from the dataframe  and replace missing values "XXX" with np.nan value. return as list.

4.get_clean_data(): Combine the two list get from (get_level_list(),get_time_list() and return a new clean data frame.
Add column names: "Date", "Water-level". 

if __name__ == '__main__':

    data1 = DataRead("insert pathname of my_data folder", "01.12.2021.csv")
    print(data1.get_clean_data())

Look similar to this.
     Time  Water_Level
0   00:15        340.0
1   00:30        340.0
2   00:45        343.0
3   01:00        340.0
4   01:15        340.0
..    ...          ...
91  23:00        340.0
92  23:15        343.0
93  23:30        343.0
94  23:45          NaN
95  23:59        343.0

[96 rows x 2 columns]

Process finished with exit code 0

5.get_average_data(): calculate the average data from a file.

if __name__ == '__main__':

    data1 = DataRead("insert pathname of my_data folder", "01.12.2021.csv")
    print(data1.get_average_data)
	
Look similar to this

337.13

Process finished with exit code 0


### modelling.py
Class Modelling. Input parameter a dataframe. Methods are described below.

1.get_train(): First 50 values are seperated as train data set.

Running this code 

if __name__ == '__main__':
    my_df = pd.read_csv("insert the pathname of  modified_average_file.csv", header=0,
                        index_col=False,
                        skipinitialspace=True,
                        skip_blank_lines=True)
    datas = Modelling(my_df)
    print(datas.get_train())

Look similar to this


         Date  Average_Water_Level
0   2021-11-19               356.09
1   2021-11-20               367.71
2   2021-11-23               337.34
3   2021-11-24               332.58
4   2021-11-25               323.83
5   2021-11-29               328.97
6   2021-11-30               343.31
7   2021-12-01               337.13
8   2021-12-02               337.13
9   2021-12-03               460.91
10  2021-12-04               487.91
11  2021-12-05               478.04
12  2021-12-06               605.44
13  2021-12-07               602.00
14  2021-12-08               536.69
15  2021-12-09               503.00
16  2021-12-10               499.72
17  2021-12-11               482.30
18  2021-12-12               467.37
19  2021-12-13               450.10
20  2021-12-14               461.39
21  2021-12-15               539.98
22  2021-12-16               577.33
23  2021-12-17               533.85
24  2021-12-18               520.15
25  2021-12-19               504.06
26  2021-12-20               484.63
27  2021-12-27               654.17
28  2021-12-28               627.62
29  2021-12-29               713.47
30  2021-12-30               965.32
31  2021-12-31              1250.31
32  2022-01-01              1296.56
33  2022-01-02              1080.64
34  2022-01-03               900.05
35  2022-01-04               846.71
36  2022-01-05               919.93
37  2022-01-06              1092.40
38  2022-01-07              1127.60
39  2022-01-08               968.71
40  2022-01-09               868.52
41  2022-01-10               774.46
42  2022-01-11               750.33
43  2022-01-12               709.67
44  2022-01-13               673.35
45  2022-01-14               645.24
46  2022-01-15               611.23
47  2022-01-16               574.36
48  2022-01-17               542.34
49  2022-01-18               535.57

Process finished with exit code 0

2.get_test(): Seperate the data frame from 51 row to the end and return as test dataset.

Run:

if __name__ == '__main__':
    my_df = pd.read_csv("insert pathname of modified_average_file.csv", header=0,
                        index_col=False,
                        skipinitialspace=True,
                        skip_blank_lines=True)
    datas = Modelling(my_df)
    print(datas.get_test())

look similar to this 
         Date  Average_Water_Level
50  2022-01-19               552.51
51  2022-01-20               545.34
52  2022-01-21               535.33
53  2022-01-22               535.06
54  2022-01-23               524.02
55  2022-01-24               523.04


3.get_ordrmse(): p,d, q value is selected for a wide range.
		self.p = range(0, 4)
        self.d = range(0, 4)
        self.q = range(0, 4)
        self.orders = []
        self.rmse = []
with iteretools a list of p,d,q is set up.
[(0, 0, 0), (0, 0, 1), (0, 0, 2)......(3, 3, 2), (3, 3, 3)]
For each p,d,q set ARIMA model is applied for train data set to predict the future values upto the length of test dataset.
Afterwards it is compared with actual test data and root mean squared errors (rmse) are calculated for each set.
Return the p,d,q set and respective errors as a dataframe.
Run:
if __name__ == '__main__':
    my_df = pd.read_csv("insert pathname of modified_average_file.csv", header=0,
                        index_col=False,
                        skipinitialspace=True,
                        skip_blank_lines=True)
    datas = Modelling(my_df)
    print(datas.get_ordrmse())

will print this

      orders        rmse
0   (3, 2, 3)    9.694525
1   (0, 1, 0)   10.591959
2   (0, 1, 1)   10.682993
3   (1, 1, 1)   10.830014
4   (0, 1, 2)   11.027566
..        ...         ...
59  (3, 3, 0)   98.495423
60  (1, 3, 3)  129.526327
61  (2, 3, 0)  144.848849
62  (1, 3, 0)  218.710151
63  (0, 3, 0)  273.950057

[64 rows x 2 columns]

Process finished with exit code 0

4.get_my_pdq(): return the order with least rmse from the dataframe.
5.get_rmse(): return the respective rmse for the chosen p,d,q set.
6.get_pred_model(): Return prediction as "pred_fit" for train data set upto the length of test data.
7.get_my_r2(): return co relation coefficient for predicted data and test data.
8. get_forecast_model(): Apply arima method with the chosen p,d,q set for total data.
run:
if __name__ == '__main__':
    my_df = pd.read_csv("insert pathname of modified_average_file.csv", header=0,
                        index_col=False,
                        skipinitialspace=True,
                        skip_blank_lines=True)
    datas = Modelling(my_df)
    print(datas.get_forecast_model())

look similar to this:

C:\Users\asheka\anaconda3\python.exe D:/pythonProject/danube/modelling.py
56    528.947969
57    527.827890
58    527.097135
59    521.790380
60    519.915409
61    516.412162
62    515.784846
63    513.064404
64    511.926265
65    508.939543
Name: predicted_mean, dtype: float64

Process finished with exit code 0

7.__str__: magic method to print the model evaluation information.





	


### classical_arima.py
Contains class Classical. Input parameter : a dataframe. With column name "Date", "Average_Water_Level". 
Method are below

1.get_timeseries(): will return a timeseries of the dataframe column "Average_Water_Level"
similar to this.

C:\Users\asheka\anaconda3\python.exe D:/pythonProject/danube/classical_arima.py
0      356.09
1      367.71
2      337.34
3      332.58
4      323.83
5      328.97
6      343.31
7      337.13
8      337.13
9      460.91
10     487.91
11     478.04
12     605.44
13     602.00
14     536.69
15     503.00
16     499.72
17     482.30
18     467.37
19     450.10
20     461.39
21     539.98
22     577.33
23     533.85
24     520.15
25     504.06
26     484.63
27     654.17
28     627.62
29     713.47
30     965.32
31    1250.31
32    1296.56
33    1080.64
34     900.05
35     846.71
36     919.93
37    1092.40
38    1127.60
39     968.71
40     868.52
41     774.46
42     750.33
43     709.67
44     673.35
45     645.24
46     611.23
47     574.36
48     542.34
49     535.57
50     552.51
51     545.34
52     535.33
53     535.06
54     524.02
55     523.04
Name: Average_Water_Level, dtype: float64
def adf_test(): Input parameter: timeseries

2.adf_test(): Input parameter : timeseries. 
 return the dicky fuller test result to check if the data is stationary or not.
 
3.@property 
    my_order(): property decorator to return the self._order value. 
	self._order = 0, 0, 0
	
4.@my_order.setter: decorator setter. Input parameter "neworder". It can be tuple or string. If not, will print error message.

5. my_train_model(): return prediction with train data upto the length of test data set. 
6.my_forecast(): return prediction for total data for the next 10 days.
7.__call__(): return rmse and corelation coefficient for actual test data and preicted data upto the length of test data set.
      


