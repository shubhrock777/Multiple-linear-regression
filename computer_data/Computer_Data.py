import pandas as pd


#loading the dataset
computer = pd.read_csv("D:/BLR10AM/Assi/21.Multiple liner regression/Datasets_MLR/Computer_Data.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Index row number (irrelevant ,does not provide useful Informatiom)",
                "Price of computer(relevant provide useful Informatiom)",
                "computer speed (relevant provide useful Informatiom)",
                "Hard Disk space of computer (relevant provide useful Informatiom)",
                "Random axis momery of computer (relevant provide useful Informatiom)",
                "Screen size of Computer (relevant provide useful Informatiom)",
                "Compact dist (relevant provide useful Informatiom)",
                "Multipurpose use or not (relevant provide useful Informatiom)",
                "Premium Class of computer (relevant provide useful Informatiom)",
                "advertisement expenses (relevant provide useful Informatiom)",
                "Trend position in market (relevant provide useful Informatiom)"]

d_types =["Count","Ratio","Ratio","Ratio","Ratio","Ratio","Binary","Binary","Binary","Ratio","Ratio"]

data_details =pd.DataFrame({"column name":computer.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": computer.dtypes})

            #3.	Data Pre-computercessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of computer 
computer.info()
computer.describe()          

#droping index colunms 
computer.drop(['Unnamed: 0'], axis = 1, inplace = True)


#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


computer['cd'] = LE.fit_transform(computer['cd'])
computer['multi'] = LE.fit_transform(computer['multi'])
computer['premium'] = LE.fit_transform(computer['premium'])

#data types        
computer.dtypes


#checking for na value
computer.isna().sum()
computer.isnull().sum()

#checking unique value for each columns
computer.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": computer.columns,
      "mean": computer.mean(),
      "median":computer.median(),
      "mode":computer.mode(),
      "standard deviation": computer.std(),
      "variance":computer.var(),
      "skewness":computer.skew(),
      "kurtosis":computer.kurt()}

EDA


# covariance for data set 
covariance = computer.cov()
covariance

# Correlation matrix 
co = computer.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphicomputer repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(computer.iloc[:, :])


#boxplot for every columns
computer.columns
computer.nunique()

computer.boxplot(column=['price','ads', 'trend'])   #no outlier

#for imputing HVO for Price column
"""
# here we can see lVO For Price
# Detection of outliers (find limits for RM based on IQR)
IQR = computer['Price'].quantile(0.75) - computer['Price'].quantile(0.25)
upper_limit = computer['Price'].quantile(0.75) + (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

computer['Price']= pd.DataFrame( np.where(computer['Price'] > upper_limit, upper_limit, computer['Price']))

import seaborn as sns 
sns.boxplot(computer.Price);plt.title('Boxplot');plt.show()"""



# Q-Q Plot
from scipy import stats
import pylab
import matplotlib.pyplot as plt

stats.probplot(computer.price, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.ads, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.trend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(computer.iloc[:,[1,2,3,4,8,9]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=computer.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = computer.iloc[:,[5,6,7]]

model_df = pd.concat([enc_df, df_norm,computer.iloc[:,[0]] ], axis =1)

#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
       
         
ml1 = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)




# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_hd = smf.ols('hd ~ speed  + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 
vif_hd # vif is low 

rsq_ram = smf.ols('ram ~ speed  + hd + screen +   cd + multi + premium + ads + trend', data = model_df).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 
vif_ram # vif is low 



 # by r squared value
mlhd = smf.ols('hd ~ speed  + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit()
mlhd.summary()

#model 2 
ml2 = smf.ols('price ~ speed + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model

# Summary
ml2.summary()

#model 3 
ml3 = smf.ols('price ~ speed + hd + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model

# Summary
ml3.summary()

# Final model
final_ml = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit()
final_ml.summary() 
final_ml.summary2()
# Prediction
pred = final_ml.predict(model_df)

from scipy import stats
import pylab
import statsmodels.api as sm
# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Jointplot
import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train, test = train_test_split(model_df, test_size = 0.2,random_state = 7) # 20% test data



# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = train).fit()

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid = test_pred - test.price
import numpy as np
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


