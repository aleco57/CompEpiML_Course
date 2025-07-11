### Day 4 - Causal Inference and ML

library(dagitty)
library(caret)
library(ranger)

# Read data
fev <- read.csv("https://biostatistics.dk/puff/data/fev.csv", header=TRUE)
fev1 <- fev
fev1$Smoke <- "Yes"

fev0 <- fev
fev0$Smoke <- "No"

#Model
model <- lm(FEV ~ Smoke + Ht + Gender, data = fev)

#Predict with the different datasets
pred1 <- predict(model, fev1)
pred0 <- predict(model, fev0)

#Calculate average of these predicted differences
mean(pred1 - pred0)

summary(model)


#### Can we get a better prediction? Lets do a random forest
model_rf <- ranger(FEV ~ Smoke + Ht + Gender, data = fev, num.trees = 100)

#Predict with the different datasets
pred1 <- predict(model_rf, fev1)$predictions
pred0 <- predict(model_rf, fev0)$predictions

#Calculate average of these predicted differences
mean(pred1 - pred0)


ctrl <- trainControl(
  method = "cv",               # Cross-validation
  number = 5,                  # 5-fold CV
  search = "grid"            # or "random"
)

# Define tuning grid
grid <- expand.grid(
  mtry = c(1,2),         # Number of variables sampled at each split
  min.node.size = c(1, 5, 10),
  splitrule = "variance"             
)

# Fit model with tuning
rf_tuned <- train(
  y = fev[,"FEV"],
  x = fev[,c("Smoke", "Ht", "Gender")],
  method = "ranger",           
  tuneGrid = grid,
  trControl = ctrl,
  num.trees = 500,
  importance = 'impurity'     
)

plot(rf_tuned)

#What are paramters for the best model?
best_params <- rf_tuned$bestTune

#Predict with the different datasets
pred1 <- predict(rf_tuned, fev1)
pred0 <- predict(rf_tuned, fev0)

#Calculate average of these predicted differences
mean(pred1 - pred0)




library(tidyverse)
library(pcalg)
library(dagitty)
library(ggdag)
library(ggplot2)
library(igraph)

