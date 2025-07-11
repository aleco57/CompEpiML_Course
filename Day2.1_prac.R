
#Lib
library(tidyverse)
library(rpart)
library(ggparty)
library(caret)
library(ranger)
library(yardstick)
library(partykit)

d_NSDUH2021 <- readRDS("d_NSDUH2021.Rdata")


##################
# Decision Tree
##################

#Construct a dataset called d_hash_drinks45 where HashEver is not NA and nor is Alc45drinks30d (see slides for the d_coc_drinks45 dataset)
d_hash_drinks45 <- d_NSDUH2021 %>%
  filter(!is.na(HashEver), !is.na(Alc45drinks30d))

#Make a classification tree with HashEver as response using rpart.
fit <- rpart(HashEver ~ ., data = d_hash_drinks45, method = "class")
par(xpd = TRUE)
plot(fit, compress = T)
text(fit, use.n = TRUE)

#Make a prettier visualisation with bar plots at the leaves using ggparty
party_fit <- as.party(fit)
ggparty(party_fit) +
  geom_edge() +
  geom_edge_label(size = 3) +
  geom_node_splitvar() +
  geom_node_label(aes(label = paste0("n = ", nodesize)),
                  ids = "terminal",
                  size = 3,
                  # 0.01 nudge_y is enough to be above the node plot since a terminal
                  # nodeplot's top (not center) is at the node's coordinates.
                  nudge_y = 0.025) +
  # pass list to gglist containing all ggplot components we want to plot for each
  # (default: terminal) node
  geom_node_plot(gglist = list(geom_bar(aes(x = "", fill = CocaineEver),
                                        position = position_fill()),  
                               xlab(NULL),
                               ylab(NULL),
                               labs(fill = "Tried cocaine?"),
                               theme_bw(),
                               theme(axis.ticks.x = element_blank())))

#Investigate whether the tree should be pruned (start with a lower cp parameter initially, e.g. cp = 0.0001)
fit2 <- rpart(HashEver ~ ., data = d_hash_drinks45, method = "class", control = rpart.control(cp = 0.0001))
par(xpd = TRUE)
plot(fit2, compress = T)
text(fit2, use.n = TRUE)

#Plot the complexity parameter (cp) table to see how the model performs with different cp values
plotcp(fit2)
plotcp(fit)

#What is the best cp value?
best_cp <- fit2$cptable[which.min(fit$cptable[, "xerror"]), "CP"]

#Now we can prune the tree
pruned_fit <- prune(fit2, cp = best_cp)

#Plot the pruned tree
par(xpd = TRUE)
plot(pruned_fit, compress = T)
text(pruned_fit, use.n = TRUE)


##################
# Random Forest
##################

d_model <- d_NSDUH2021 |> 
  select(Age, Gender, MaritalStatus, CountyType, Education, WorkLastWeek,
         Sad30d, Hopeless30d, Nerv30d, 
         PatientMH12m, Patient12m, 
         AlcEver, SmokeEver, SmklssTobEver, HashEver, CocaineEver,
         Smoke30d, Smoke100tms, 
         Alc45drinks30d, HashUsed30d) |> 
  na.omit()
nrow(d_model)

fit <- rpart(HashEver ~ ., d_model)
fit2 <- rpart(HashEver ~ ., d_model, control = rpart.control(cp = 0.0001))

#Now fit a random forest with 100 trees
rf_fit <- ranger(HashEver ~ ., data = d_model, num.trees = 100, probability = T)

rf_probs <- predict(rf_fit, data = d_model)$predictions
# Convert probabilities to class labels
rf_pred <- colnames(rf_probs)[max.col(rf_probs)]
rf_pred <- factor(rf_pred, levels = levels(d_model$HashEver))
rf_confusion <- confusionMatrix(data = rf_pred, reference = d_model$HashEver, mode = "sens_spec")
print(rf_confusion)

#Compare this confusion matrix with fit 2
# Make a confusion matrix for the pruned decision tree
fit2_probs <- predict(fit2, newdata = d_model)
fit2_pred <- colnames(fit2_probs)[max.col(fit2_probs)]
fit2_pred <- factor(fit2_pred, levels = levels(d_model$HashEver))

fit2_confusion <- confusionMatrix(data = fit2_pred, reference = d_model$HashEver, mode = "sens_spec")
print(fit2_confusion)

#Make an ROC curve for both the random forest and the pruned decision tree
d_model_pred <- tibble(Truth = d_model |> pull(HashEver), 
                       rf = rf_probs[,1],
                       prune_dt = fit2_probs[,1])
d_rocs <- bind_rows(
  roc_curve(d_model_pred, Truth, rf) |> mutate(model = "rf"),
  roc_curve(d_model_pred, Truth, prune_dt) |> mutate(model = "prune_dt")
)
ggplot(d_rocs, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal()
#Calculate the AUC for both models
rf_auc <- roc_auc(d_model_pred, Truth, rf)
prune_dt_auc <- roc_auc(d_model_pred, Truth, prune_dt)
print(rf_auc)
print(prune_dt_auc)


### Now we can use the ALS dataset to comapre rf with LASSO regression
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
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1)

#Make prediction accuracy from the test dataset using the 1se lambda value
lasso_pred <- predict(lasso_cv, newx = x_test, s = "lambda.1se")
#Calculate the MSE
lasso_mse <- mean((y_test - lasso_pred)^2)

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
  splitrule = "variance"             
)

# Fit model with tuning
rf_tuned <- train(
  x = x_train,
  y = y_train,
  method = "ranger",           
  tuneGrid = grid,
  trControl = ctrl,
  num.trees = 500,
  importance = 'impurity'     
)

plot(rf_tuned)

#What are paramters for the best model?
best_params <- rf_tuned$bestTune

# Make predictions on the test set
rf_pred <- predict(rf_tuned, newdata = x_test)
# Calculate the MSE
rf_mse <- mean((y_test - rf_pred)^2)
# Compare the MSE of the lasso and random forest models
print(paste("Lasso MSE:", lasso_mse))
print(paste("Random Forest MSE:", rf_mse))




##########################
# Random Forest continued
###########################

#Tune hyperparamters with ranger using d_model data and HashEver as outcome
rf_tuned <- train(
  HashEver ~ ., 
  data = d_model, 
  method = "ranger", 
  trControl = trainControl(method = "cv", number = 5, classProbs = T, summaryFunction = multiClassSummary),
  tuneGrid = expand.grid(
    mtry = c(2, 4, 6, 8),         
    min.node.size = c(1, 5, 10),
    splitrule = "gini"             
  ),
  num.trees = 10,
  metric = "logLoss"
)

plot(rf_tuned)

