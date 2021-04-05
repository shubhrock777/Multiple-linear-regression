import pandas as pd


#loading the dataset
toyo = pd.read_csv("D:/BLR10AM/Assi/21.Multiple liner regression/Datasets_MLR/ToyotaCorolla.csv", encoding ="latin1")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary




data_details =pd.DataFrame({"column name":toyo.columns,
                            "data type(in Python)": toyo.dtypes})

            #3.	Data Pre-toyocessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of toyo 
toyo.info()
toyo.describe()          

#droping index colunms 
toyo.drop(["Id"], axis = 1, inplace = True)


#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


toyo['Age_08_04'] = LE.fit_transform(toyo['Age_08_04'])
toyo['HP'] = LE.fit_transform(toyo['HP'])
toyo['cc'] = LE.fit_transform(toyo['cc'])
toyo['Doors'] = LE.fit_transform(toyo['Doors'])
toyo['Gears'] = LE.fit_transform(toyo['Gears'])

df= toyo[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

#data types        
df.dtypes


#checking for na value
df.isna().sum()
df.isnull().sum()

#checking unique value for each columns
df.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

    


EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA


# covariance for data set 
covariance = df.cov()
covariance

# Correlation matrix 
co = df.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])


#boxplot for every columns
df.columns
df.nunique()

df.boxplot(column=['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax', 'Weight'])   #no outlier





#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(df.iloc[:,1:9])
df.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=df.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns



model_df = pd.concat([df,toyo.iloc[:,[1]] ], axis =1)

#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model


         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

model_df_new = model_df.drop(model_df.index[[960,221]])
#droping row num 960,221 due to outlire

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df_new).fit()    

# Summary
ml_new.summary()

# Prediction
pred = ml_new.predict(model_df)


# removing outlier 


# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(model_df)

from scipy import stats
import pylab
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Q-Q plot  residuals
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Jointplot
import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
pro_train, pro_test = train_test_split(model_df_new, test_size = 0.2,random_state = 77) # 20% test data



# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = pro_train).fit()
model_train.summary()
# prediction on test data set 
test_pred = model_train.predict(pro_test)

import numpy as np
# test residual values 
test_resid = test_pred - pro_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(pro_train)

# train residual values 
train_resid  = train_pred - pro_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


