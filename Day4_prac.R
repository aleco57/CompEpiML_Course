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




library(dagitty)
library(ggdag)
library(ggplot2)
library(igraph)
#install.packages('pcalg','ggdag','igraph', 'causaldisco')
library(tidyverse)
 
# Load and or read in the data
d_NSDUH2021 <- readRDS("~/Downloads/Computational_Biology/d_NSDUH2021.Rdata")
 
### Data preparation
 
d_model <- d_NSDUH2021 %>% 
  filter(MaritalStatus == "Married") %>% # filter by marrital status
  select(Gender, Education, 
         AlcEver, SmokeEver,  HashEver, CocaineEver, HashUsed30d) |> 
  na.omit()
d_model
 
set.seed(1)
x <- d_model %>% 
  #sample_n(1000) %>% # Do not sample
  mutate_all(as.character) %>% mutate_all(as.factor) %>% 
                mutate_all(as.integer) %>% 
                mutate_all(~ . - 1L)
#head(x)
#head(nlev)
nlev <- unname(apply(x,2,function(x) length(unique(x))))
suf_stat <- list(dm=x,nlev=nlev,adaptDF=FALSE)
 
##### Run the PC algorithm
cpdag <- pc(suffStat = suf_stat,
            indepTest = disCItest,
            alpha = 0.01, # Vary the alpha to test which edges needs to be removed
            labels = colnames(x)
  )
### Plot the model
ig <- graph_from_graphnel(cpdag@graph)
el <- igraph::as_edgelist(ig)
G <- dagitty(paste0('dag{',paste0(apply(el,1,function(x) paste0(x[1],"->",x[2])),
                                  collapse = '\n'),
                    "}"))
G |> tidy_dagitty(layout = 'kk')|> ggdag() + theme_dag()

