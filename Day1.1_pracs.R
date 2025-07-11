library(tidyverse)
library(splines)
library(boot)
library(caret)

# Set to data dir
setwd("/Users/am17168/Library/CloudStorage/OneDrive-UniversityofBristol/PhD/Training/Courses/EEPE/Computational_Epi/Data")
# Load the data
data <- readRDS("d_NSDUH2021.Rdata")


###############
# Section 1, data exploration
###############


#Make bar plot for HashEver
ggplot(data, aes(x = HashEver)) +
  geom_bar() +
  labs(title = "Bar Plot of HashEver", x = "HashEver", y = "Count") +
  theme_minimal()

# Make bar plot for HashEver stratified by Gender
ggplot(data, aes(x = HashEver, fill = Gender)) +
  geom_bar(position = "stack") +
  labs(title = "Bar Plot of HashEver", x = "HashEver", y = "Count") +
  theme_minimal()

#Investigate if there is an association between SmokeTryAge and HashTryAge for those that have tried both alcohol and hash.
lm(HashTryAge ~ SmokeTryAge, data = filter(data, SmokeEver == "Yes", HashEver == "Yes")) %>%
  summary()

#Lets also plot this data
ggplot(filter(data, SmokeEver == "Yes", HashEver == "Yes"), aes(x = SmokeTryAge, y = HashTryAge)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Scatter Plot of SmokeTryAge vs HashTryAge",
       x = "Smoke Try Age",
       y = "Hash Try Age") +
  theme_minimal()

#investigate the distribution of other demographic variables like gender and education.
ggplot(data, aes(x = Gender)) +
  geom_bar() +
  labs(title = "Bar Plot of Gender", x = "Gender", y = "Count") +
  theme_minimal()

ggplot(data, aes(x = Education)) +
  geom_bar() +
  labs(title = "Bar Plot of Education", x = "Education", y = "Count") +
  theme_minimal()

#What is the missingness of each variable?
missingness <- sapply(data, function(x) sum(is.na(x)))
missingness_df <- data.frame(Variable = names(missingness), MissingCount = missingness)
ggplot(missingness_df, aes(x = Variable, y = MissingCount)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Missingness of Variables", x = "Variable", y = "Missing Count") +
  theme_minimal()

#Missingness can be important, we may want to impute these values as cc could be biased


###############
# Section 2, CV
###############

d_smoke_alc <- data |> 
  filter(AlcEver == "Yes", SmokeEver == "Yes",
         !is.na(SmokeTryAge), !is.na(AlcTryAge))
m_lm <- lm(AlcTryAge ~ SmokeTryAge, d_smoke_alc)

#Does a spline model using 5 splines have better out of model prediction than linear regression?
#Set up the train_control object

train_control <- trainControl(method = "repeatedcv", 
                              number = 4,
                              repeats = 10)

set.seed(123)
# Train the linear model
lm_model <- train(AlcTryAge ~ SmokeTryAge, 
                  data = d_smoke_alc, 
                  method = "lm", 
                  trControl = train_control)
# Train the spline model with 5 splines
spline_model <- train(AlcTryAge ~ ns(SmokeTryAge, df = 5), 
                       data = d_smoke_alc, 
                       method = "lm", 
                       trControl = train_control)

# Compare the models, lets first look at the linear performance
mean(lm_model$resample$RMSE)
mean(lm_model$resample$Rsquared)
# Compare the models, lets first look at the spline performance
mean(spline_model$resample$RMSE)
mean(spline_model$resample$Rsquared)

#Plot these predictions into figure
ggplot(d_smoke_alc, aes(x = SmokeTryAge, y = AlcTryAge)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  geom_smooth(aes(y = predict(spline_model, newdata = d_smoke_alc)), color = "red", se = FALSE) +
  labs(title = "Comparison of Linear and Spline Models",
       x = "Smoke Try Age",
       y = "Alcohol Try Age") +
  theme_minimal() +
  scale_color_manual(values = c("blue" = "Linear Model", "red" = "Spline Model"))


#######################
# Section 3, Bootstrap
#######################
#Make a non-parametric bootstrap (e.g. using the boot package)
model_coef_boot <- function(data, index) {
  coef(lm(AlcTryAge ~ SmokeTryAge, data = data, subset = index))
}

b <- boot(data = d_smoke_alc, 
          statistic = model_coef_boot, 
          R = 1000)

#Make a histogram of the bootstrap disribution
ggplot(data.frame(boot_coef = b$t[, 2]), aes(x = boot_coef)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Bootstrap Distribution of SmokeTryAge Coefficient",
       x = "Coefficient Value",
       y = "Frequency") +
  theme_minimal()

#Compare bootstrap standard errors with that from coef(summary(m_lm))
boot_se <- sd(b$t[, 2])
lm_se <- summary(m_lm)$coefficients[2, 2]
data.frame(
  Method = c("Bootstrap", "Linear Model"),
  StandardError = c(boot_se, lm_se)
)

#Compare 95%CIs
quantile(b$t[, 2], c(0.025, 0.975))
confint(m_lm)



#Now lets do the same but make a parametric bootstrap



#########################
# Section 3, Loss Scores
#########################

d_cooper <- read.table("data-cooper-infered.csv", sep = ";", header = TRUE) |> 
  as_tibble()

m0 <- lm(y ~ x, data = d_cooper)

#Implement the mean squared loss and compare to the parameter estimates from coef(m0)
loss_mse <- function(par) {
  
  alpha <- par[1]
  beta1 <- par[2]
  pred_y <- alpha + beta1*d_cooper$x
  
  mean((d_cooper$y - pred_y)^2)
}

#Now use absolute loss
loss_abs <- function(par) {
  
  alpha <- par[1]
  beta1 <- par[2]
  pred_y <- alpha + beta1*d_cooper$x
  
  mean(abs(d_cooper$y - pred_y))
}


my_mse_fit <- optim(par = c(0, 0), fn = loss_mse)
my_abs_fit <- optim(par = c(0, 0), fn = loss_abs)


coef(summary(m0))



