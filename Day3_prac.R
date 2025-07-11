#### ML Comp


#Packages
library(ggplot2)
library(tidyverse)
library(caret)
library(rpart)
library(ggparty)
library(caret)
library(ranger)
library(yardstick)
library(partykit)
library(glmnet)
library(keras)
library(tensorflow)
library(torch)
library(torchvision)
library(scorcher)
library(xgboost)
library(ParBayesianOptimization)
library(SuperLearner)



######################
# First load the data
######################

load("data/andata.rda")


#######################
# Clean the data
#######################

#There are a lot of dummy variables where we only have a small number of people in one of the categories.
# Quantify this and remove those with very low power
dummy_vars <- apply(traindata_x, 2, function(x) length(unique(x)) == 2) %>%
  which() %>%
  names()

#Now make a variable for the continuous variables
continuous_vars <- apply(traindata_x, 2, function(x) length(unique(x)) > 2) %>%
  which() %>%
  names()

# Check how many people are in each category
dummy_counts <- sapply(dummy_vars, function(var) {
  table(testdata_x[[var]])
}, simplify = FALSE)

#Extract those with low counts
low_count_flags <- sapply(dummy_counts, function(tbl) any(tbl < 10))
sparse_vars <- names(low_count_flags[low_count_flags])


#Lets remove these from our train dataset
traindata_x_clean <- traindata_x %>%
  select(-any_of(sparse_vars))

#Make a heat map of all the correlations of all the continuous predictors
correlation_matrix <- cor(traindata_x_clean, use = "pairwise.complete.obs")
# Plot the heatmap
heatmap(correlation_matrix, 
        main = "Correlation Heatmap of Predictors and DEATH2YRS",
        col = colorRampPalette(c("blue", "white", "red"))(100),
        scale = "column")

#Lets onlt take a single predictor if highly correlted with other variables
highly_correlated_vars <- findCorrelation(correlation_matrix, cutoff = 0.75)
# Remove highly correlated variables from the dataset
traindata_x_clean <- traindata_x_clean[, -highly_correlated_vars]

#Now make our test dataset clean removing the same variables
testdata_x_clean <- testdata_x %>%
  select(any_of(names(traindata_x_clean)))


################################
# Correlations with the training
################################

#Now we can run a ttest to see if the predictors are significantly different between the two groups in the outcome
#Create a function to perform t-tests for each variable
ttest_results <- sapply(traindata_x_clean[,continuous_vars], function(var) {
  t.test(var ~ traindata_DEATH2YRS, data = cbind(traindata_DEATH2YRS, var))$p.value
})
# Convert the results to a data frame for better readability
ttest_results_df <- data.frame(
  Variable = names(ttest_results),
  P_Value = ttest_results
)

#Lets also run a chi squared for our binary variables to see which are associated
chi_squared_results <- apply(traindata_x[,dummy_vars], 2, function(var) {
  chisq.test(table(traindata_DEATH2YRS, var))$p.value
})

# Convert the results to a data frame for better readability
chi_squared_results_df <- data.frame(
  Variable = names(chi_squared_results),
  P_Value = chi_squared_results
)

#Now lets extract the variables which were bonferonni sig (0.05 / ncol(traindata_x_clean))
bonferroni_threshold <- 0.05 / ncol(traindata_x_clean)
significant_vars <- ttest_results_df[ttest_results_df$P_Value < bonferroni_threshold, "Variable"]
# Combine significant continuous and dummy variables
significant_dummy_vars <- chi_squared_results_df[chi_squared_results_df$P_Value < bonferroni_threshold, "Variable"]
# Combine both sets of significant variables
significant_vars_combined <- c(significant_vars, significant_dummy_vars)

#################
# Model Building
#################

#Lets write a function which will predict the outcome using a model and return the accuracy compared with test data
predictFunction <- function(testdata_x, model) {
  
  #choose the variables from newdata that we will use:
  usedata <- testdata_x
  
  #Use prediction function for glm models:
  probabilities <- predict(model, newdata = usedata, type = "response")
  
  #Convert probabilities to binary predictions
  predictions <- ifelse(probabilities > 0.5, 1, 0)
  
  #Calculate confusion matrix so we know accuracy
  confusion_matrix <- table(predictions, testdata_DEATH2YRS)
  
  #Calculate accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  #return the accuracy
  return(accuracy)
}

#Make a dataframe where we can add the results of each model
model_results <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  stringsAsFactors = FALSE
)


####################################
# Start with logistic regression
####################################

#PSA is assocaited with prostate cancer survival, lets do a logistic regression of this var along with age
set.seed(570)
psa_model <- glm(traindata_DEATH2YRS ~ PSA + AGEGRP_65to74 + AGEGRP_75plus,
                 data = cbind(traindata_DEATH2YRS, traindata_x_clean),
                 family = binomial)
# Get the summary of the model
summary(psa_model)

#Predict and add to df
model_results <- rbind(model_results, data.frame(Model = "PSA_Age Logistic Model", 
                                                  Accuracy = predictFunction(testdata_x, psa_model)))

#Lets now do a model with our significant variables
set.seed(570)
significant_model <- glm(traindata_DEATH2YRS ~ ., 
                          data = cbind(traindata_DEATH2YRS, traindata_x_clean[, significant_vars_combined]),
                          family = binomial)
# Get the summary of the model
summary(significant_model)

model_results <- rbind(model_results, data.frame(Model = "Significant Variables Logistic Model", 
                                                  Accuracy = predictFunction(testdata_x, significant_model)))

#Lets make another model also including psa and age with the sig vars
set.seed(570)
full_model <- glm(traindata_DEATH2YRS ~ .,
                   data = cbind(traindata_DEATH2YRS, traindata_x_clean[, c(significant_vars_combined, "PSA", "AGEGRP_65to74", "AGEGRP_75plus")]),
                   family = binomial)
# Get the summary of the model
summary(full_model)

#Predict and add to df
model_results <- rbind(model_results, data.frame(Model = "Sig_vars, PSA and Age Logistic Model", 
                                                  Accuracy = predictFunction(testdata_x, full_model)))



########################
# ElasticNet regression
########################

#Make the outcome a factor so we can run logistic regression 
traindata_DEATH2YRS_f <- factor(traindata_DEATH2YRS, levels = c(0, 1))
levels(traindata_DEATH2YRS_f) <- c("No", "Yes")

#Lets tune both the alpha and lambda parameters for the elastic net
grid <- expand.grid(
  alpha = seq(0, 1, 0.1),
  lambda = 10^seq(-4, 1, length = 100)
)

#Change the following code so it is logistic elastic net not linear regression:
ctrl <- trainControl(method = "cv", number = 10, classProbs = T, summaryFunction = defaultSummary)

set.seed(570)
en_cv_model <- train(
  x = traindata_x_clean,
  y = traindata_DEATH2YRS_f,
  method = "glmnet",
  family = "binomial",  
  tuneGrid = grid,
  trControl = ctrl,
  metric = "Accuracy"
)

plot(en_cv_model)

#Check the best alpha and lambda
best_alpha <- en_cv_model$bestTune$alpha
best_lambda <- en_cv_model$bestTune$lambda
#Fit the elastic net model with the best parameters
elastic_net_fit <- glmnet(
  as.matrix(traindata_x_clean),
  traindata_DEATH2YRS_f,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "binomial"
)

#Now get model predictions
predictions_en <- predict(elastic_net_fit, newx = as.matrix(testdata_x_clean), type = "response")
# Convert probabilities to binary predictions
predicted_class_en <- as.integer(predictions_en > 0.5)
#Calculate confusion matrix
confusion_matrix_en <- table(testdata_DEATH2YRS, predicted_class_en)
#Calculate accuracy
accuracy_en <- sum(diag(confusion_matrix_en)) / sum(confusion_matrix_en)
#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Elastic Net Model (clean df)", 
                                                  Accuracy = accuracy_en))



## Now run a second elastic net model, but always keep PSA and two age variables in the model by having a penality factor
set.seed(570)
penalty <- rep(1, ncol(traindata_x_clean))
names(penalty) <- colnames(traindata_x_clean)
penalty[c("PSA", "AGEGRP_65to74", "AGEGRP_75plus")] <- 0 

#Fit the elastic net model with the penalty factor
elastic_net_pen_model <- glmnet(
  as.matrix(traindata_x_clean),
  traindata_DEATH2YRS_f,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "binomial",
  penalty.factor = penalty
)


#Now get model predictions
predictions_en_pen <- predict(elastic_net_pen_model, newx = as.matrix(testdata_x_clean), type = "response")
# Convert probabilities to binary predictions
predicted_class_en_pen <- as.integer(predictions_en_pen > 0.5)
#Calculate confusion matrix
confusion_matrix_en_pen <- table(testdata_DEATH2YRS, predicted_class_en_pen)
#Calculate accuracy
accuracy_en_pen <- sum(diag(confusion_matrix_en_pen)) / sum(confusion_matrix_en_pen)
#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Elastic Net Model with Penalty (clean df)", 
                                                  Accuracy = accuracy_en_pen))



#######################
# Run an random forest
#######################

#Now we can fit a random forest model tuning the number of trees
ctrl <- trainControl(
  method = "cv",               # Cross-validation
  number = 5,                  # 5-fold CV
  search = "grid"            # or "random"
)

# Define tuning grid
grid <- expand.grid(
  mtry = c(2, 4, 6, 8),         # Number of variables sampled at each split
  min.node.size = c(1, 5, 10),
  splitrule = "gini"             
)

# Fit model with tuning
set.seed(570)
rf_tuned <- train(
  x = traindata_x_clean,
  y = traindata_DEATH2YRS_f,
  method = "ranger",           
  tuneGrid = grid,
  trControl = ctrl,
  num.trees = 500,
  importance = 'impurity',
  metric = "Accuracy"
)

plot(rf_tuned)

#What are paramters for the best model?
best_params <- rf_tuned$bestTune

#Fit the random forest model with the best parameters
rf_model <- ranger(
  formula = traindata_DEATH2YRS_f ~ ., 
  data = cbind(traindata_DEATH2YRS_f, traindata_x_clean),
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  splitrule = best_params$splitrule,
  num.trees = 500,
  importance = 'impurity',
  probability = TRUE
)

#Now we can predict with the model
predictions_rf <- predict(rf_model, data = testdata_x_clean)$predictions
#Convert probabilities to binary predictions
rf_pred <- colnames(predictions_rf)[max.col(predictions_rf)]
#Convert to factor for comparison
rf_pred <- factor(rf_pred, levels = c("No", "Yes"))
#Calculate confusion matrix
confusion_matrix_rf <- table(testdata_DEATH2YRS, rf_pred)
#Calculate accuracy
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Random Forest Model (clean df)", 
                                                  Accuracy = accuracy_rf))


#######################
# Now try with XGBOOST
#######################
set.seed(570)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  search = "grid",
  classProbs = TRUE,                # enable probability predictions
  summaryFunction = defaultSummary # or twoClassSummary if you want ROC
)

grid <- expand.grid(
  nrounds = c(100, 200),           # number of boosting iterations
  max_depth = c(3, 6, 9),          # max depth of tree
  eta = c(0.01, 0.1, 0.3),         # learning rate
  gamma = 0,                       # minimum loss reduction
  colsample_bytree = 0.8,          # subsample ratio of columns
  min_child_weight = 1,            # minimum sum of instance weight in a child
  subsample = 0.8                  # subsample ratio of training data
)

xgb_tuned <- caret::train(
  x = traindata_x_clean,
  y = traindata_DEATH2YRS_f,
  method = "xgbTree",
  tuneGrid = grid,
  trControl = ctrl,
  metric = "Accuracy"
)

# Plot the tuning results
plot(xgb_tuned)
# Get the best parameters
best_xgb_params <- xgb_tuned$bestTune
# Fit the XGBoost model with the best parameters
xgb_model <- xgboost(
  data = as.matrix(traindata_x_clean),
  label = as.numeric(traindata_DEATH2YRS_f) - 1, # Convert factor to numeric (0 and 1)
  nrounds = best_xgb_params$nrounds,
  max_depth = best_xgb_params$max_depth,
  eta = best_xgb_params$eta,
  gamma = best_xgb_params$gamma,
  colsample_bytree = best_xgb_params$colsample_bytree,
  min_child_weight = best_xgb_params$min_child_weight,
  subsample = best_xgb_params$subsample,
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = FALSE
)

#Now we can predict with the model
predictions_xgb <- predict(xgb_model, newdata = as.matrix(testdata_x_clean))
# Convert probabilities to binary predictions
predicted_class_xgb <- as.integer(predictions_xgb > 0.5)
#Calculate confusion matrix
confusion_matrix_xgb <- table(testdata_DEATH2YRS, predicted_class_xgb)
#Calculate accuracy
accuracy_xgb <- sum(diag(confusion_matrix_xgb)) / sum(confusion_matrix_xgb)

#Add to model results
model_results <- rbind(model_results, data.frame(Model = "XGBoost Model (clean df)", 
                                                  Accuracy = accuracy_xgb))


#Now we can try again, this time will try and tune the hyperparameters better
#Prep the data for xgboost
x_mat <- as.matrix(traindata_x_clean)
y_vec <- as.numeric(traindata_DEATH2YRS_f) - 1  # convert factor to 0/1

dtrain <- xgb.DMatrix(data = x_mat, label = y_vec)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",       # classification error (lower is better)
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  gamma = 1
)

set.seed(570)
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 20,
  maximize = FALSE,            # we are minimizing error
  verbose = 1
)

best_iter <- cv_model$best_iteration
lowest_error <- cv_model$evaluation_log$test_error_mean[best_iter]
best_accuracy <- 1 - lowest_error

cat("Best iteration:", best_iter, "\n")
cat("Best cross-validated accuracy:", round(best_accuracy, 4), "\n")

final_model <- xgboost(
  data = dtrain,
  params = params,
  nrounds = best_iter,
  verbose = 1
)

#Now we can predict with the final model
predictions_final_xgb <- predict(final_model, newdata = as.matrix(testdata_x_clean))
# Convert probabilities to binary predictions
predicted_class_final_xgb <- as.integer(predictions_final_xgb > 0.5)
#Calculate confusion matrix
confusion_matrix_final_xgb <- table(testdata_DEATH2YRS, predicted_class_final_xgb)
#Calculate accuracy
accuracy_final_xgb <- sum(diag(confusion_matrix_final_xgb)) / sum(confusion_matrix_final_xgb)

#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Final XGBoost Model (clean df)", 
                                                  Accuracy = accuracy_final_xgb))




#### Now we can try a bayseian tuning
set.seed(570)
scoringFunction <- function(eta, max_depth, subsample, colsample_bytree) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",  # XGBoost's error = 1 - accuracy
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    verbosity = 0
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # xgb.cv uses "error", so we subtract from 1 to get accuracy
  best_accuracy <- 1 - min(cv$evaluation_log$test_error_mean)
  
  list(Score = best_accuracy)
}

bounds <- list(
  eta = c(0.01, 0.3),
  max_depth = c(3L, 10L),
  subsample = c(0.6, 1),
  colsample_bytree = c(0.6, 1)
)

opt_obj <- bayesOpt(
  FUN = scoringFunction,
  bounds = bounds,
  initPoints = 10,     # Random starts
  iters.n = 30,        # Bayesian steps
  acq = "ucb",         # Acquisition function
  verbose = 1
)

# 1. Extract best parameters from the Bayesian optimization object
best_params <- getBestPars(opt_obj)

# 2. Convert your data into xgb.DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(traindata_x_clean), label = traindata_DEATH2YRS)
dtest <- xgb.DMatrix(data = as.matrix(testdata_x_clean), label = testdata_DEATH2YRS)

# 3. Define full parameter list
final_params <- list(
  objective = "binary:logistic",
  eval_metric = "error",  # Optimize for accuracy
  eta = best_params$eta,
  max_depth = as.integer(best_params$max_depth),
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  verbosity = 0
)

# 4. Train final model — use best nrounds from CV or pick a large number + early stopping
xgb_final <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

#Now we can predict with the final model
predictions_final_xgb_bayes <- predict(xgb_final, newdata = as.matrix(testdata_x_clean))
# Convert probabilities to binary predictions
predicted_class_final_xgb_bayes <- as.integer(predictions_final_xgb_bayes > 0.5)
#Calculate confusion matrix
confusion_matrix_final_xgb_bayes <- table(testdata_DEATH2YRS, predicted_class_final_xgb_bayes)
#Calculate accuracy
accuracy_final_xgb_bayes <- sum(diag(confusion_matrix_final_xgb_bayes)) / sum(confusion_matrix_final_xgb_bayes)

#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Final XGBoost Model with Bayesian Tuning (clean df)", 
                                                  Accuracy = accuracy_final_xgb_bayes))



## Lets do the same bayseian tuning but this time with the full dataset not the clean
#Make a DMatrix for the full dataset
dtrain_full <- xgb.DMatrix(data = as.matrix(traindata_x), label = traindata_DEATH2YRS)
dtest_full <- xgb.DMatrix(data = as.matrix(testdata_x), label = testdata_DEATH2YRS)

# Define the scoring function for Bayesian optimization
set.seed(570)
scoringFunction <- function(eta, max_depth, subsample, colsample_bytree) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",  # XGBoost's error = 1 - accuracy
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    verbosity = 0
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain_full,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # xgb.cv uses "error", so we subtract from 1 to get accuracy
  best_accuracy <- 1 - min(cv$evaluation_log$test_error_mean)
  
  list(Score = best_accuracy)
}

bounds <- list(
  eta = c(0.01, 0.3),
  max_depth = c(3L, 10L),
  subsample = c(0.6, 1),
  colsample_bytree = c(0.6, 1)
)

opt_obj <- bayesOpt(
  FUN = scoringFunction,
  bounds = bounds,
  initPoints = 10,     # Random starts
  iters.n = 30,        # Bayesian steps
  acq = "ucb",         # Acquisition function
  verbose = 1
)

# 1. Extract best parameters from the Bayesian optimization object
best_params_full <- getBestPars(opt_obj)
# 2. Define full parameter list
final_params_full <- list(
  objective = "binary:logistic",
  eval_metric = "error",  # Optimize for accuracy
  eta = best_params_full$eta,
  max_depth = as.integer(best_params_full$max_depth),
  subsample = best_params_full$subsample,
  colsample_bytree = best_params_full$colsample_bytree,
  verbosity = 0
)

# 3. Train final model — use best nrounds from CV or pick a large number + early stopping
xgb_final_full <- xgb.train(
  params = final_params_full,
  data = dtrain_full,
  nrounds = 200,
  watchlist = list(train = dtrain_full, test = dtest_full),
  early_stopping_rounds = 10,
  print_every_n = 10
)

#Now we can predict with the final model
predictions_final_xgb_full <- predict(xgb_final_full, newdata = as.matrix(testdata_x))
# Convert probabilities to binary predictions
predicted_class_final_xgb_full <- as.integer(predictions_final_xgb_full > 0.5)
#Calculate confusion matrix
confusion_matrix_final_xgb_full <- table(testdata_DEATH2YRS, predicted_class_final_xgb_full)
#Calculate accuracy
accuracy_final_xgb_full <- sum(diag(confusion_matrix_final_xgb_full)) / sum(confusion_matrix_final_xgb_full)
#Add to model results
model_results <- rbind(model_results, data.frame(Model = "Final XGBoost Model with Bayesian Tuning (full df)", 
                                                  Accuracy = accuracy_final_xgb_full))


########################
# NeuralNet regression
########################

#First will need to prep and scale the data for NN
#Scale data 
NN_traindata_x <- scale(traindata_x)

colMeansTrain <- attr(NN_traindata_x, "scaled:center")
colSDsTrain <- attr(NN_traindata_x, "scaled:scale")
NN_testdata_x <- scale(testdata_x, center = colMeansTrain, 
                       scale = colSDsTrain)

#Convert x data to matrices:
NN_traindata_x <- as.matrix(NN_traindata_x)
NN_testdata_x <- as.matrix(NN_testdata_x)

#Convert outcomes to "one hot deck" encoding:
NN_traindata_DEATH2YRS <- model.matrix(~ factor(traindata_DEATH2YRS) - 1)
NN_traindata_DISCONT <- model.matrix(~ factor(traindata_DISCONT) - 1)
NN_testdata_DEATH2YRS <- model.matrix(~ factor(testdata_DEATH2YRS) - 1)
NN_testdata_DISCONT <- model.matrix(~ factor(testdata_DISCONT) - 1)

#Convert to tensors
x_train_tens <- torch_tensor(NN_traindata_x, dtype = torch_float())
y_train_tens <- torch_tensor(NN_traindata_DEATH2YRS[,2], dtype = torch_float())

# Now we can run the first NN model
set.seed(570)
dl <- scorch_create_dataloader(x_train_tens, y_train_tens, batch_size = 50)

scorch_model <- dl |> 
  initiate_scorch() |>
  scorch_layer("dropout", p=0.1) |>   # Dropout layer to prevent overfitting
  scorch_layer("linear", 91, 91) |>    # Layer 1
  scorch_layer("linear", 91, 1) |>     # Output layer
  scorch_layer("sigmoid")              # Activation 1

compiled_scorch_model <- compile_scorch(scorch_model)

fitted_scorch_model <- compiled_scorch_model |> 
  fit_scorch(
    loss = nn_bce_loss,            # How wrong are we? Not the LOSS function
    num_epochs = 100,               # 20 full passes through the data
    verbose = TRUE                 # Show training progress
  )
#Now we can predict with the model
predictions_scorch <- fitted_scorch_model(NN_testdata_x) %>% as.numeric()
predicted_class <- as.integer(predictions_scorch > 0.5)

table(testdata_DEATH2YRS, predicted_class)

#Calculate accuracy
accuracy_scorch <- sum(testdata_DEATH2YRS == predicted_class) / length(testdata_DEATH2YRS)


### Looping of hyperparamters for tuning
hidden_sizes <- c(10, 50)
dropouts <- c(0.0, 0.1)
batch_sizes <- c(10, 50)
#Tune size for second and third layer
hidden_sizes2 <- c(10, 50)
hidden_sizes3 <- c(10, 50)
#Tune size for dropout 2 and 3
dropouts2 <- c(0.0, 0.1)
dropouts3 <- c(0.0, 0.1)

results <- list()

set.seed(570)
for (h in hidden_sizes) {
  for (d in dropouts) {
    for (b in batch_sizes) {
      for (h2 in hidden_sizes2) {
        for (d2 in dropouts2) {
          for (h3 in hidden_sizes3) {
            for (d3 in dropouts3) {
              # Print current hyperparameters
              cat("Training model with:", h, "units,", d, "dropout,", b, "batch size,", 
                  h2, "units layer 2,", d2, "dropout layer 2,", 
                  h3, "units layer 3,", d3, "dropout layer 3\n")
              
              # Create data loader
              dl <- scorch_create_dataloader(x_train_tens, y_train_tens, batch_size = b)
              
              # Define model
              model <- dl |>
                initiate_scorch() |>
                scorch_layer("dropout", p = d) |>
                scorch_layer("linear", ncol(x_train_tens), h) |>
                scorch_layer("dropout", p = d2) |>
                scorch_layer("linear", h, h2) |>
                scorch_layer("dropout", p = d3) |>
                scorch_layer("linear", h2, h3) |>
                scorch_layer("linear", h3, 1) |>
                scorch_layer("sigmoid")
              
              compiled <- compile_scorch(model)
              
              fitted <- compiled |>
                fit_scorch(
                  loss = nn_bce_loss,
                  num_epochs = 30,
                  verbose = TRUE  
                )
              
              # Evaluate (e.g., accuracy)
              predictions_scorch <- fitted(NN_testdata_x) %>% as.numeric()
              predicted_class <- as.integer(predictions_scorch > 0.5)
              acc <- sum(testdata_DEATH2YRS == predicted_class) / length(testdata_DEATH2YRS)
              
              results[[paste(h, d, b, h2, d2, h3, d3, sep = "_")]] <- acc
            }
          }
        }
      }
    }
  }
}
      


# Convert results to a data frame for better readability
results_df <- data.frame(
  Hyperparameters = names(results),
  Accuracy = unlist(results)
)


#Order by accuracy
results_df <- results_df %>%
  arrange(desc(Accuracy))




######################################################
# Now we can run a super learner with the best models
######################################################

#First we need to make a list of the models we want to use

#Set seed
set.seed(570)

# Create a SuperLearner object
sl_model <- SuperLearner(
  Y = traindata_DEATH2YRS,
  X = traindata_x_clean,
  family = binomial(),
  SL.library = c("SL.glm", "SL.glmnet", "SL.ranger", "SL.xgboost"),
  method = "method.NNLS"
)

# Now predict from the model
sl_predictions <- predict(sl_model, newdata = testdata_x_clean)$pred
# Convert probabilities to binary predictions
sl_predicted_class <- as.integer(sl_predictions > 0.5)
# Calculate confusion matrix
confusion_matrix_sl <- table(testdata_DEATH2YRS, sl_predicted_class)
# Calculate accuracy
accuracy_sl <- sum(diag(confusion_matrix_sl)) / sum(confusion_matrix_sl)



