import pandas as pd


#loading the dataset
startup = pd.read_csv("D:/BLR10AM/Assi/21.Multiple liner regression/Datasets_MLR/50_Startups.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Money spend on research and development",
                "Administration",
                "Money spend on Marketing",
                "Name of state",
                "Company profit"]

d_types =["Ratio","Ratio","Ratio","Nominal","Ratio"]

data_details =pd.DataFrame({"column name":startup.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": startup.dtypes})

            #3.	Data Pre-startupcessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of startup 
startup.info()
startup.describe()          


#rename the columns
startup.rename(columns = {'R&D Spend':'rd_spend', 'Marketing Spend' : 'm_spend'} , inplace = True)  

#data types        
startup.dtypes


#checking for na value
startup.isna().sum()
startup.isnull().sum()

#checking unique value for each columns
startup.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": startup.columns,
      "mean": startup.mean(),
      "median":startup.median(),
      "mode":startup.mode(),
      "standard deviation": startup.std(),
      "variance":startup.var(),
      "skewness":startup.skew(),
      "kurtosis":startup.kurt()}

EDA


# covariance for data set 
covariance = startup.cov()
covariance

# Correlation matrix 
co = startup.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphistartup repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(startup.iloc[:, :])


#boxplot for every columns
startup.columns
startup.nunique()

startup.boxplot(column=['rd_spend', 'Administration', 'm_spend', 'Profit'])   #no outlier

# here we can see lVO For profit
# Detection of outliers (find limits for RM based on IQR)
IQR = startup['Profit'].quantile(0.75) - startup['Profit'].quantile(0.25)
lower_limit = startup['Profit'].quantile(0.25) - (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

startup['Profit']= pd.DataFrame( np.where(startup['Profit'] < lower_limit, lower_limit, startup['Profit']))

import seaborn as sns 
sns.boxplot(startup.Profit);plt.title('Boxplot');plt.show()



# rd_spend
plt.bar(height = startup.rd_spend, x = np.arange(1, 51, 1))
plt.hist(startup.rd_spend) #histogram
plt.boxplot(startup.rd_spend) #boxplot


# Administration
plt.bar(height = startup.Administration, x = np.arange(1, 51, 1))
plt.hist(startup.Administration) #histogram
plt.boxplot(startup.Administration) #boxplot

# m_spend
plt.bar(height = startup.m_spend, x = np.arange(1, 51, 1))
plt.hist(startup.m_spend) #histogram
plt.boxplot(startup.m_spend) #boxplot




#profit
plt.bar(height = startup.Profit, x = np.arange(1, 51, 1))
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot


# Jointplot

sns.jointplot(x=startup['Profit'], y=startup['rd_spend'])



# Q-Q Plot
from scipy import stats
import pylab

stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show() 
# startupfit is normally distributed

stats.probplot(startup.Administration, dist = "norm", plot = pylab)
plt.show() 
# administration is normally distributed


stats.probplot(startup.rd_spend, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(startup.m_spend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup.iloc[:,[0,1,2]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=startup.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = pd.get_dummies(startup.iloc[:,[3]])
enc_df.columns
enc_df.rename(columns={"State_New York":'State_New_York'},inplace= True)

model_df = pd.concat([enc_df, df_norm, startup.iloc[:,4]], axis =1)

#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model


         
ml1 = smf.ols('Profit ~ State_California+ State_Florida+ State_New_York+ rd_spend + Administration + m_spend ', data = model_df).fit() # regression model

# Summary
ml1.summary2()
ml1.summary()
# p-values for State, Administration are more th no correlation of model_dffit with State and Administrationan 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm 


sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

model_df_new = model_df.drop(model_df.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~State_California+ State_Florida+ State_New_York+ rd_spend + Administration + m_spend ', data = model_df_new).fit()    

# Summary
ml_new.summary()
ml_new.summary2()


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd_spend = smf.ols('rd_spend ~ Administration + m_spend + State_California+ State_Florida+ State_New_York', data = model_df).fit().rsquared  
vif_rd_spend = 1/(1 - rsq_rd_spend) 

rsq_admini = smf.ols(' Administration ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York ', data = model_df).fit().rsquared  
vif_admini = 1/(1 - rsq_admini)
ml_ad=smf.ols(' Administration ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York ', data = model_df).fit()
ml_ad.summary() 

rsq_m_spend = smf.ols(' m_spend ~ rd_spend + Administration  + State_California+ State_Florida+ State_New_York', data = model_df).fit().rsquared  
vif_m_spend = 1/(1 - rsq_m_spend) 

rsq_state = smf.ols(' State_California ~ rd_spend + Administration + m_spend  ', data = model_df).fit().rsquared  
vif_state = 1/(1 - rsq_state) 

ml_S= smf.ols(' State_California~ rd_spend + Administration + m_spend  ', data = model_df).fit()
ml_S.summary()


# Storing vif values in a data frame
d1 = {'Variables':['rd_spend' ,'Administration' ,'m_spend ',' State '], 'VIF':[vif_rd_spend, vif_admini, vif_m_spend, vif_state]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#vif is low 

#model 2 without Administration  column because p value of state >>> 0.5

ml2 = smf.ols('Profit ~ rd_spend + State_California+ State_Florida+ State_New_York + m_spend  ', data = model_df).fit() # regression model

# Summary
ml2.summary()

# administration p value is high 

sm.graphics.influence_plot(ml2)

# Studentized Residuals = Residual/standard deviation of residuals



#model 3 without Administration  and state column because p value of state >>> 0.5
ml3 = smf.ols('Profit ~ rd_spend + m_spend ', data = model_df_new).fit() # regression model

# Summary
ml3.summary()

sm.graphics.influence_plot(ml3)



# Final model
final_ml = smf.ols('Profit ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York  ', data = model_df).fit() 
final_ml.summary() 

# Prediction
pred = final_ml.predict(model_df)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data



# preparing the model on train data 
model_train = smf.ols('Profit ~ rd_spend + m_spend  + State_California+ State_Florida+ State_New_York  ', data = model_df_train).fit()
model_train.summary()
model_train.summary2()
# prediction on test data set 
test_pred = model_train.predict(model_df_test)

# test residual values 
test_resid = test_pred - model_df_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(model_df_train)

# train residual values 
train_resid  = train_pred - model_df_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

