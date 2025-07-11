#libs
library(dplyr)
library(glmnet)
library(MESS)
library(hal9001)
library(stabs)

#Load in data
als <- read.table("https://hastie.su.domains/CASI_files/DATA/ALS.txt", header=TRUE)

#Generate our test dataset and extract the outcome and predictors
als_train <- als[als$testset==FALSE, ]
y_train <- als_train$dFRS
x_train <- als_train[,-c(1:2)] %>% as.matrix()

#Do the same for test
als_test <- als[als$testset==TRUE, ]
y_test <- als_test$dFRS
x_test <- als_test[,-c(1:2)] %>% as.matrix()

#Fit a lasso model
lasso_fit <- glmnet(x_train, y_train, alpha = 1)
#Plot the lasso fit
plot(lasso_fit, xvar = "lambda", label = TRUE)

#The model must be standardised but this is done by the model. This is important as a penality is added to the coefficients
#If the data was not standardised then the coefficients would be penalised based on its units, rather than contribution to the model.
# I.e. a variable with a large range would be penalised more than a variable with a small range, even if the contribution to the model was the same.

#Now lets do a cross validation to find the best lambda
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1)

plot(lasso_cv)

#Extract the non-zero predictors from the model
lasso_coef <- coef(lasso_cv, s = "lambda.min")

#Now let use our fitted model on the test dataset
lasso_pred <- predict(lasso_cv, newx = x_test, s = "lambda.min")
#Calculate the MSE
mean((y_test - lasso_pred)^2)

#Extract the non-zero predictors from the cv model
delasso <- lm(y_train ~ ., data = as.data.frame(x_train[, lasso_coef@i]))

#Compare MSPE from delasso and cv lasso
mean((y_test - predict(delasso, newdata = as.data.frame(x_test[, lasso_coef@i])))^2)

#The lasso performs better than the delassoed

#Now run using ridge regression
ridge_fit <- cv.glmnet(x_train, y_train, alpha = 0)
plot(ridge_fit, xvar = "lambda", label = TRUE)

#Predict values
ridge_pred <- predict(ridge_fit, newx = x_test, s = ridge_fit$lambda.min)
mean((y_test - ridge_pred)^2)

#Now run cross validation for elastic net to find the best values
elastic_net_cv <- cv.glmnet(x_train, y_train, alpha = 0.5)

en_pred <- predict(elastic_net_cv, newx = x_test, s = "lambda.min")
#Calculate the MSE
mean((y_test - en_pred)^2)


#Lets tune both the alpha and lambda parameters for the elastic net
grid <- expand.grid(
  alpha = seq(0, 1, 0.1),
  lambda = 10^seq(-4, 1, length = 100)
)

ctrl <- trainControl(method = "cv", number = 10)

en_cv_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = grid,
  trControl = ctrl
)

#Check the best alpha and lambda
best_alpha <- en_cv_model$bestTune$alpha
best_lambda <- en_cv_model$bestTune$lambda
#Fit the elastic net model with the best parameters
elastic_net_fit <- glmnet(x_train, y_train, alpha = best_alpha, lambda = best_lambda)

#How many non zero coefficients?
length(coef(elastic_net_fit)[coef(elastic_net_fit) != 0])



#Predict values
en_pred_tuned <- predict(elastic_net_fit, newx = x_test)
#Calculate the MSE
mean((y_test - en_pred_tuned)^2)


#Now we can fit an adaptive lasso
weights <- adaptive.weights(x_train, y_train)
weights$weights[is.infinite(weights$weights) | is.na(weights$weights)] <- 1e6 
adaptive_lasso_fit <- glmnet(x_train, y_train, penalty.factor = weights$weights)

al_pred <- predict(adaptive_lasso_fit, newx = x_test)
mean((y_test - al_pred)^2)

#Run a group lasso for those variables starting with "Symptom."
group_vars <- grep("^Symptom\\.", colnames(x_train), value = TRUE)
group_indices <- as.integer(factor(ifelse(colnames(x_train) %in% group_vars, "Symptom", "Other")))
group_lasso_fit <- glmnet(x_train, y_train, alpha = 0, group = group_indices)
