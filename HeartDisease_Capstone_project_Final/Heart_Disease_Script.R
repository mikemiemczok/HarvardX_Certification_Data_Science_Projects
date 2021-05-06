############ Do your own project - Capstone - Predicting Heart disease. ##########################

# Installing packages and loading libraries.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

library(caret)
library(rpart)
library(corrplot)
library(naniar)
library(tidyverse)
library(tidyr)
library(caret)
library(data.table)
library(knitr)
library(lubridate)


# Downloading the heart csv from github.

dl <- tempfile()
download.file("https://raw.githubusercontent.com/mikemiemczok/HarvardX_Certification_Data_Science_Projects/main/HeartDisease_Capstone_project_Final/heart.csv", dl)
heart <- read.csv(dl)

# The goal of the project it is to predict heart disease based on the dataset.

# Getting an overview over the structure of the dataset
str(heart)
# Age
# sex
# cp -> chest pain type
# trestbps -> resting blood pressure
# chol -> cholestoral
# fbs -> fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false
# restecg -> resting electrocardiographic results
# thalach -> maximum heart rate achieved
# exang <- exercise induced angina (1 = yes, 0 = no)
# oldpeak <- ST depression induced by exercise relative to rest
# slope <- the slope of the peak exercise ST segment
# ca <- number of major vessels (0-3) colored by flourosopy
# thal <- 1 = fixed defect; 2 = normal; 3 = reversable defect
# target <- Heart disease 0 = yes; 1 = no.

# Number of columns and rows.
nrow(heart)
ncol(heart)

# Checking the dataset for N/As
sapply(heart, function(x){
  sum(is.na(x))
})

# Starting the explorative data analysis with some visualizations
# Getting an overview over the predicting column
table(heart$target)
# The dataset seems to be balanced. 138 of the observations had heart disease and 165 had no heart disease.

# Preparing the dataset for visualization. Changing the numerical values of sex into human readable.
heart_visual <- heart %>% mutate(sex = ifelse(sex == 1, "Male", "Female"))

# Starting with some visualization to get an overview of the observations.
heart_visual %>% 
  group_by(sex) %>%
  summarise(count = n()) %>%
  ggplot(aes(sex, count, fill = sex)) +
  geom_col() +
  geom_text(aes(label = count, x = sex, y = count), size = 5, colour = "White", vjust = 1.5) +
  labs(title = "General distribution of gender",
       x = "Gender",
       y = "Prevelance")

# 0 -> Heart Disease. 1 -> no heart disease.
heart_visual %>% 
  group_by( target, sex) %>%
  summarise(count = n()) %>%
  ggplot(aes(ifelse(target == 0, "Disease", "No disease"), count, fill = sex)) +
  geom_col() +
  labs(title = "Distribution of gender having the disease and being healthy",
       x = "Target",
       y = "Prevelance")

# CP = Chest Pain (0: typical angina; 1: atypical angina; 2: non-anginal pain; 3: asymptomatic)
heart_visual %>% 
  group_by(cp, target) %>%
  summarise(count = n()) %>%
  ggplot(aes(cp, count, fill = factor(target))) +
  geom_col() +
  labs(title = "Distribution of chest pain types having the disease and being healthy",
       x = "Chest Pain Type",
       y = "Prevelance", fill = "Disease")

# Thalium Stress Test Result (1 = fixed defect; 2 = normal; 3 = reversable defect)
heart_visual %>% 
  filter(thal != 0) %>%
  group_by(thal, target) %>%
  summarise(count = n()) %>%
  ggplot(aes(thal, count, fill = factor(target))) +
  geom_col() +
  labs(title = "Distribution of thalium stress test result having the disease and being healthy",
       x = "Thalium Stress Test Result",
       y = "Prevelance", fill = "Disease")

# Exercise Induced Angina (0 = no; 1 = yes)

heart_visual %>% 
  group_by(exang, target) %>%
  summarise(count = n()) %>%
  ggplot(aes(factor(exang), count, fill = factor(target))) +
  geom_col() +
  labs(title = "Distribution of exercise induced angina having the disease and being healthy",
       x = "Exercise Induced Angina",
       y = "Prevelance", fill = "Disease")

# Fasting Blood Sugar > 120 mg/dl (0 = no; 1 = yes)
heart_visual %>% 
  group_by(fbs, target) %>%
  summarise(count = n()) %>%
  ggplot(aes(fbs, count, fill = factor(target))) +
  geom_col() +
  labs(title = "Distribution of high fasting blood sugar having the disease and being healthy",
       x = "Fasting Blood Sugar > 120mg/dl",
       y = "Prevelance", fill = "Disease")


# Inspecting the Age Effect on Heart Disease
heart_visual %>% 
  group_by(ï..age, target) %>%
  summarise(count = n()) %>%
  ggplot(aes(ï..age, count , fill = factor(target), colour = factor(target))) +
  geom_col() +
  labs(title = "General distribution of the age having a heart disease or not",
       x = "Age",
       y = "Prevelance", fill = "Disease", colour = "Disease")

# Changing the dataset heart_visual to the dataset heart, because it's better to building models with numerical values.
# Building the datasets to train and validate with.
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = heart$target, times = 1, p = 0.2, list = FALSE)
work_with <- heart[-test_index,]
validate_with <- heart[test_index,]

# Splitting the Train set into a train/test set.
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = work_with$target, times = 1, p = 0.2, list = FALSE)
train_set <- work_with[-test_index,]
test_set <- work_with[test_index,]

# Building the Models.
# Testing multiple models on the dataset.
models <- c("glm", "lda", "rpart", "naive_bayes", "svmLinear", "knn", "rf")

fits <- lapply(models, function(model){ 
  print(model)
  train(factor(target) ~ ., method = model, data = train_set)
}) 

names(fits) <- models

glm_results <- confusionMatrix(predict(fits$glm, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- data_frame(Variables = "GLM in general", Accuracy = glm_results)

lda_results <- confusionMatrix(predict(fits$lda, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "LDA in general", Accuracy = lda_results))

rpart_results <- confusionMatrix(predict(fits$rpart, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "RPART in general", Accuracy = rpart_results))

nb_results <- confusionMatrix(predict(fits$naive_bayes, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "Naive bayes in general", Accuracy = nb_results))

svm_results <- confusionMatrix(predict(fits$svmLinear, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "SVM in general", Accuracy = svm_results))

knn_results <- confusionMatrix(predict(fits$knn, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "KNN in general", Accuracy = knn_results))

rf_results <- confusionMatrix(predict(fits$rf, test_set), factor(test_set$target))$overall["Accuracy"]
model_results <- bind_rows(model_results, data_frame(Variables = "RF in general", Accuracy = rf_results))

model_results %>% knitr::kable()

# Based on the results we decide to take the best three performing models - Naive Bayes, Random Forest, Generalized Linear Model.
# Now we are deploying the three mentioned models again with specific parameters.

# Naive bayes
# Using crossvalidation to get the optimal values for the tuneGrids laplace and adjust.
lap <- seq(0, 6, 0.5)
adj <- seq(1,7, 0.5)
  
train_nb2 <- sapply(lap, function(l){
  train <- train(factor(target) ~ .,
                    method = "naive_bayes",
                    tuneGrid = data.frame(usekernel = TRUE, laplace = l, adjust = adj),
                    data = train_set)
  train$results
})

accuracy_nb <- sapply(seq(1, 13, 1), function(i){
  max(train_nb2["Accuracy",i]$Accuracy)
})

laplace_nb <- sapply(seq(1, 13, 1), function(i){
  train_nb2["laplace",i]$laplace[which.max(train_nb2["Accuracy",i]$Accuracy)]
})

adjust_nb <- sapply(seq(1, 13, 1), function(i){
  train_nb2["adjust",i]$adjust[which.max(train_nb2["Accuracy",i]$Accuracy)]
})

# Visualizing the accuracy depending laplace and adjust.
df_nb <- data.frame(Accuracy = accuracy_nb, Laplace = laplace_nb, Adjust = adjust_nb)
df_nb %>% ggplot(aes(Laplace, Accuracy, colour = factor(Adjust))) + geom_point()
# The visualization shows us that the best tuning parameters for adjust and lapace are: laplace = 3.5 and adjust = 2.5


# Fitting the optimzed naive bayes model and prediciting.
train_nb <- train(factor(target) ~ .,
                  method = "naive_bayes",
                  tuneGrid = data.frame(usekernel = TRUE, laplace = 3.5, adjust = 2.5),
                  data = train_set)

nb_results_after_optimization <- confusionMatrix(predict(train_nb, test_set), factor(test_set$target))$overall["Accuracy"]
optimized_model_results <- data_frame(Variables = "Naive Bayes after optimization", Accuracy = nb_results_after_optimization)


# Random Forest
train_rf2 <- train(factor(target) ~ .,
                 method = "rf",
                 tuneGrid = data.frame(mtry = seq(1, 5, 0.5)),
                 data = train_set)

# Visualizing parameter mtry against the accuracy to get the best mtry value.
plot(train_rf2)

# Fitting and deploying the optimized rf model.
train_rf <- train(factor(target) ~ .,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 2),
                  data = train_set)

rf_results_after_optimization <- confusionMatrix(predict(train_rf, test_set), factor(test_set$target))$overall["Accuracy"]
optimized_model_results <- bind_rows(optimized_model_results, data_frame(Variables = "Random forest after optimization", Accuracy = rf_results_after_optimization))


# Generalized Logistic Regression
train_glm <- train(factor(target) ~ .,
                   method = "glm",
                   data = train_set)
glm_results_after_optimization <- confusionMatrix(predict(train_glm, test_set), factor(test_set$target))$overall["Accuracy"]
optimized_model_results <- bind_rows(optimized_model_results, data_frame(Variables = "GLM after optimization", Accuracy = glm_results_after_optimization))

optimized_model_results %>% knitr::kable()

# Checking existing correlations on the whole heart dataset to improve the models by deleting non correlation columns.
correlations <- cor(heart)
corrplot(correlations, "number")

# As we can see in the graphic there are 4 columns with a correlation under 0.15.
# Assuming that this columns haven't any positive impact on the model we a are deleting them to get a better result.
train_set$chol <- NULL
train_set$fbs <- NULL
train_set$trestbps <- NULL
train_set$restecg <- NULL

# Deploying the three models again.
train_nb <- train(factor(target) ~ .,
                  method = "naive_bayes",
                  tuneGrid = data.frame(usekernel = TRUE, laplace = 0.5, adjust = 2.5),
                  data = train_set)
nb_optimized_corr <- confusionMatrix(predict(train_nb, test_set), factor(test_set$target))$overall["Accuracy"]
corr_optimized_model_results <- data_frame(Variables = "Naive Bayes after optimization - correlation", Accuracy = nb_optimized_corr)


# Random Forest
train_rf <- train(factor(target) ~ .,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 2),
                  data = train_set)
rf_optimized_corr <- confusionMatrix(predict(train_rf, test_set), factor(test_set$target))$overall["Accuracy"]
corr_optimized_model_results <- bind_rows(corr_optimized_model_results, data_frame(Variables = "Random Forest after optimization - correlation", Accuracy = rf_optimized_corr))

# Generalized Logistic Regression
train_glm <- train(factor(target) ~ .,
                   method = "glm",
                   data = train_set)
glm_optimized_corr <- confusionMatrix(predict(train_glm, test_set), factor(test_set$target))$overall["Accuracy"]
corr_optimized_model_results <- bind_rows(corr_optimized_model_results, data_frame(Variables = "GLM after optimization - correlation", Accuracy = glm_optimized_corr))

# Trying to use ensemble to get the best out of all models.
p_nb <- predict(train_nb, test_set)
p_rf <- predict(train_rf, test_set)
p_glm <- predict(train_glm, test_set)
p <- as.numeric(p_nb) + as.numeric(p_rf) + as.numeric(p_glm)
y_pred <- factor(ifelse(p > 4, 1, 0))

ensemble_optimized_corr <- confusionMatrix(y_pred, factor(test_set$target))$overall["Accuracy"]
corr_optimized_model_results <- bind_rows(corr_optimized_model_results, data_frame(Variables = "Ensemble", Accuracy = ensemble_optimized_corr))

corr_optimized_model_results %>% knitr::kable()

# We see that the deleting of the columns chol, fbs, trestbps and restecg had an impact on the model result.
# Also we can see that naive bayes is performing better than any other model and better as the combination of the three models.

# Deleting columns with nearly zero correlation.
validate_with$chol <- NULL
validate_with$fbs <- NULL
validate_with$trestbps <- NULL
validate_with$restecg <- NULL

# Building the final model - naive bayes - on the validation dataset.
# Naive bayes
final_nb <- confusionMatrix(predict(train_nb, validate_with), factor(validate_with$target))$overall["Accuracy"]
final_model_results <- data_frame(Variables = "Naive Bayes - final model", Accuracy = final_nb)
final_model_results %>% knitr::kable()