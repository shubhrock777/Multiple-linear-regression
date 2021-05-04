av_ds <- read.csv(file.choose())
av_ds$X <- NULL #This auto created column, we don't required this
head(av_ds)


str(av_ds)

#Dependant Variable : AveragePrice, since this is continous variable so start with MLR
#Step 1: Model Validation: HOLD OUT, divide the data into train and test data, and create model on train_data
library(caret)

index <- createDataPartition(av_ds$AveragePrice, p=0.8, list = F)
train_data <- av_ds[index,]
test_data  <- av_ds[-index,]

av_model_train <- lm(AveragePrice~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)


#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)


#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)


#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)


#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Volume+type+year, data = train_data)
summary(av_model_train)

#some region factor levels are not significant but then also we keep this factor, because it also contains the significant levels
summary(av_model_train)

#Now all the variables are significant
#Step 2 : Check for MultiColinearity
library(car)
 
vif(av_model_train)


#From the Output above, Date and year are insignificant variables, first remove the variable with highest vif value that is year and re-run the model
av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)


#Re-Run Step 2 : Check for MultiColinearity
vif(av_model_train)
 



#Create the fitted and resi variables in train_data
train_data$fitt <- round(fitted(av_model_train),2)
train_data$resi <- round(residuals(av_model_train),2)
head(train_data)



#Step 3 : Checking the normality of error i.e. resi column from train_data
#There are 2 ways of doing this, as below :
#(a)lillieTest from norTest() package

library(nortest)
lillie.test(train_data$resi) #We have to accept H0: it is normal


#(b)qqplot
qqnorm(train_data$resi)
qqline(train_data$resi, col = "green")


#From graph also this is not normal
#For MLR model error must be normal, lets do some trnsformaation ton achieve this
#Step 4 : Check for Influencing data in case on non- normal error
#4.(a)
influ <- influence.measures(av_model_train)
#influ
#check for cook.d column and if any value > 1 then remove that value and re-run the model
#4.(b)
influencePlot(av_model_train, id.method = "identical", main = "Influence Plot", sub = "Circle size")


#Remove 5486 index data from the data set and re-run the model
train_data$fitt <- NULL
train_data$resi <- NULL
train_data <- train_data[-(5485),]


av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)


train_data$fitt <- round(fitted(av_model_train),2)
train_data$resi <- round(residuals(av_model_train),2)
head(train_data)
 
#Repeat 4.(b)
influencePlot(av_model_train, id.method = "identical", main = "Influence Plot", sub = "Circle size")



#Step 5 : Check for Heteroscadicity, H0 : error are randomly spread, we have to accept H0, i.e p-value must be > than 0.05
#(a)plot
plot(av_model_train)

#5.(b) ncvTest
ncvTest(av_model_train, ~Large_Bags+XLarge.Bags+type+region)

# p = 0 it means there is problem of heteroscadicity 
#Since error are not normal and there is issue of Heteroscadicity, we can transform the dependent variable and this may be resolve these issues.
#take log of Y varibale and re-run the model

train_data$fitt <- NULL
train_data$resi <- NULL
train_data$AveragePrice <- log(train_data$AveragePrice)

av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

train_data$fitt <- round(fitted(av_model_train),2)
train_data$resi <- round(residuals(av_model_train),2)
head(train_data)


#Check again, repeat Step 3 again:
lillie.test(train_data$resi)
  

qqnorm(train_data$resi)
qqline(train_data$resi, col = "green")


#Still error are not normal
ncvTest(av_model_train, ~Large_Bags+XLarge.Bags+type+region)

#Still there is issue of Heteroscadicity
#Now re-run the model and re-run the model implementation steps done above.
av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

#Checking the Nomality of error again
train_data$fitt <- fitted(av_model_train)
train_data$resi <- residuals(av_model_train)

lillie.test(train_data$resi) #way 1
 
#Again H0, ois rejected, and errors are not normal again after the transformation
qqnorm(train_data$resi)   #way 2
qqline(train_data$resi, col = "green")#way 2


#Lest Check the stability of Model using RMSE of Train and Test Data
library(ModelMetrics)

test_data$AveragePrice <- log(test_data$AveragePrice)
test_data$fitt <- predict(av_model_train, test_data)
test_data$resi <- test_data$AveragePrice - test_data$fitt

head(test_data)

RMSE_train <- RMSE(train_data$AveragePrice, train_data$fitt)
RMSE_test <- RMSE(test_data$AveragePrice, test_data$fitt)

check_stability <- paste0(  round((RMSE_test - RMSE_train)*100,2)," %")

RMSE_train

RMSE_test

check_stability 

# Since the Difference between Test and Train RMSE is less than 10%, so that the model is stable, but not linear acceptable model.
# To make the model Good, we require add more VARIABLES or PREDICTORS, so that the Adjusted R square value must be above 65% or .65