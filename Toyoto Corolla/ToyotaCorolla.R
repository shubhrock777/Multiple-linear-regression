
######################Q3
###prepare a prediction model for predicting Price.
ToyotaCorolla <- read.csv(file.choose())



Corolla <- ToyotaCorolla[, c('Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight')]

# Getting Summary of data
summary(Corolla)

attach(Corolla)

# Variance


var(Price)



var(KM)



sd(Price)


sd(KM)






colnames(Corolla)



Corolla_Model <- lm(Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight,data = Corolla)

summary(Corolla_Model)


vif(Corolla_Model)

avPlots(Corolla_Model)

stepAIC(Corolla_Model)

Corolla_Model_final <- lm(Price ~ Age_08_04+KM+HP+log(cc)+Gears+Quarterly_Tax+Weight,data = Corolla)

summary(Corolla_Model_final)

# Data Partitioning
n <- nrow(Corolla)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- Corolla[-train, ]

# Model Training
model <- lm(Price ~ Age_08_04+KM+HP+log(cc)+Gears+Quarterly_Tax+Weight,data = Corolla)
summary(model)


pred <- predict(model, newdata = test)
actual <- test$Price
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

