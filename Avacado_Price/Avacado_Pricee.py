import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


#loading the avacadoset
avacado = pd.read_csv("D:/BLR10AM/Assi/21.Multiple liner regression/Datasets_MLR/Avacado_Price.csv")

#2.	Work on each feature of the avacadoset to create a avacado dictionary as displayed in the below image
#######feature of the avacadoset to create a avacado dictionary

#######feature of the avacadoset to create a avacado dictionary




avacado_details =pd.DataFrame({"column name":avacado.columns,
                            "avacado type(in Python)": avacado.dtypes})

            #3.	avacado Pre-avacadocessing
          #3.1 avacado Cleaning, Feature Engineering, etc
          
          
#details of avacado 
avacado.info()
avacado.describe()          

#avacado types        
avacado.dtypes


#checking for na value
avacado.isna().sum()
avacado.isnull().sum()

#checking unique value for each columns
avacado.nunique()


#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


avacado['type'] = LE.fit_transform(avacado['type'])
avacado['year'] = LE.fit_transform(avacado['year'])





"""	Exploratory avacado Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": avacado.columns,
      "mean": avacado.mean(),
      "median":avacado.median(),
      "mode":avacado.mode(),
      "standard deviation": avacado.std(),
      "variance":avacado.var(),
      "skewness":avacado.skew(),
      "kurtosis":avacado.kurt()}

EDA


# covariance for avacado set 
covariance = avacado.cov()
covariance

# Correlation matrix 
co = avacado.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphiavacado repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(avacado.iloc[:, :])


#boxplot for every columns
avacado.columns
avacado.nunique()

avacado.boxplot(column=['AveragePrice', 'Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3',
       'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge Bags'])   #no outlier

#for imputing HVO for  columns
"""
# here we can see lVO For Price
# Detection of outliers (find limits for RM based on IQR)
IQR = avacado['Price'].quantile(0.75) - avacado['Price'].quantile(0.25)
upper_limit = avacado['Price'].quantile(0.75) + (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

avacado['Price']= pd.avacadoFrame( np.where(avacado['Price'] > upper_limit, upper_limit, avacado['Price']))

import seaborn as sns 
sns.boxplot(avacado.Price);plt.title('Boxplot');plt.show()"""



# Q-Q Plot
from scipy import stats
import pylab
import matplotlib.pyplot as plt


#normal

# Normalization function using z std. all are continuous avacado.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized avacado frame (considering the numerical part of avacado)
df_norm = norm_func(avacado.iloc[:,[1,2,3,4,5,6,7,8,10]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=avacado.iloc[:,[-1]]
enc_df = pd.avacadoFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns
enc_df=pd.get_dummies(avacado.iloc[:,[-1]])


model_df = pd.concat([avacado.iloc[:,[0]] , df_norm], axis =1)

model_df.rename(columns={'XLarge Bags':'XLarge_Bags'},inplace =True)
#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled avacado (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the avacado and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
       
       
ml1 = smf.ols('AveragePrice~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year', data = model_df).fit() # regression model

# Summary
ml1.summary()

# Checking whether avacado has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)




# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_Total_Bags = smf.ols('Total_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+year', data = model_df).fit().rsquared  
vif_Total_Bags = 1/(1 - rsq_Total_Bags) 
vif_Total_Bags # vif is high

rsq_Small_Bags = smf.ols('Small_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Large_Bags+XLarge_Bags+year', data = model_df).fit().rsquared  
vif_Small_Bags = 1/(1 - rsq_Small_Bags) 
vif_Small_Bags # vif is high

rsq_Large_Bags = smf.ols('Large_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+XLarge_Bags+year', data = model_df).fit().rsquared  
vif_Large_Bags = 1/(1 - rsq_Large_Bags) 
vif_Large_Bags # vif is high 

rsq_XLarge_Bags = smf.ols('XLarge_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+year', data = model_df).fit().rsquared  
vif_XLarge_Bags = 1/(1 - rsq_XLarge_Bags) 
vif_XLarge_Bags # vif is high


# Storing vif values in a data frame
d1 = {'Variables':['Total_Bags' ,'Small_Bags' ,'Large_Bags',' XLarge_Bags '], 'VIF':[vif_Total_Bags, vif_Small_Bags, vif_Large_Bags, vif_XLarge_Bags]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame



#now we building model without Total_Bags+Small_Bags+Large_Bags+XLarge_Bags
 # by r squared value
ml_1 = smf.ols('AveragePrice~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+year', data = model_df).fit()
ml_1.summary()

#not good model 

avacado.columns

avacado.drop([ 'Total_Volume', 'Total_Bags', 'region', 'year'], axis = 1,inplace = True)

scaler = StandardScaler().fit(avacado)
avacado_avocado_scaler = scaler.transform(avacado)
avacado_avocado = pd.DataFrame(avacado_avocado_scaler)
avacado_avocado.columns = ['AveragePrice', 'Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type']
avacado_avocado.head()

feature_cols = ['Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type']
X = avacado_avocado[feature_cols]

y = avacado_avocado.AveragePrice
from sklearn.model_selection import train_test_split
def split(X,y):
    return train_test_split(X, y, test_size=0.20, random_state=1)

X_train, X_test, y_train, y_test=split(X,y)
print('Train cases as below')
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)


def linear_reg( X, y, gridsearch = False):
    
    X_train, X_test, y_train, y_test = split(X,y)
    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    if not(gridsearch):
        linreg.fit(X_train, y_train) 

    else:
        from sklearn.model_selection import GridSearchCV
        parameters = {'normalize':[True,False], 'copy_X':[True, False]}
        linreg = GridSearchCV(linreg,parameters, cv = 10)
        linreg.fit(X_train, y_train)                                                           # fit the model to the training avacado (learn the coefficients)
        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  
        
        y_pred_test = linreg.predict(X_test)                                                   # make predictions on the testing set

        RMSE_test = (metrics.mean_squared_error(y_test, y_pred_test))                          # compute the RMSE of our predictions
        print('RMSE for the test set is {}'.format(RMSE_test))

    return linreg

linreg = linear_reg(X,y)

linreg.score(X,y)

print('Intercept:',linreg.intercept_)                                           # print the intercept 
print('Coefficients:',linreg.coef_)


feature_cols.insert(0,'Intercept')
coef = linreg.coef_.tolist()
coef.insert(0, linreg.intercept_)

eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)
    
y_pred_train = linreg.predict(X_train)    

y_pred_test = linreg.predict(X_test)

MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)

print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))


MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)

print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))



import numpy as np
RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))

print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))



print("Model Evaluation for Linear Regression Model")


yhat = linreg.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print("r_squared for train avacado ",r_squared, " and adjusted_r_squared for train avacado",adjusted_r_squared)


yhat = linreg.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r_squared for test avacado ",r_squared, " and adjusted_r_squared for test avacado",adjusted_r_squared)

feature_cols = ['Small', 'SmallBags', 'Type']
X1 = avacado_avocado[feature_cols]  
y1 = avacado_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)

feature_cols = ['Large', 'LargeBags', 'Type']
X1 = avacado_avocado[feature_cols]  
y1 = avacado_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)

feature_cols = ['XLarge', 'XLargeBags', 'Type']
X1 = avacado_avocado[feature_cols]  
y1 = avacado_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)

