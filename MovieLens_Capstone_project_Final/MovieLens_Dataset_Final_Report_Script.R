##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(knitr)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################## DATA PREPROCESSING ###########################

########### Add Column timeformat based on TIMESTAMP into an interpretable time format (yyyy-mm-dd hh:mm:ss UTC). #############

edx <- edx %>% mutate(timeformat = as_datetime(timestamp))
validation <- validation %>% mutate(timeformat = as_datetime(timestamp))

######### Add column monthyear based on timeformat which contains the year and month (yyyy-mm). ########

edx <- edx %>% mutate(monthyear = paste(month(timeformat),year(timeformat),sep = "-"))
validation <- validation %>% mutate(monthyear = paste(month(timeformat),year(timeformat),sep = "-"))


########### Seperate Genres. E.g. (Action/Romantic) into (Action) (Romantic). ##############

edx <- edx %>% separate_rows(genres, sep = "\\|")
validation <- validation %>% separate_rows(genres, sep = "\\|")

########################### Visualisation #############################

######## Overview between Months and Ratings ########

edx %>%
  mutate(month = month(timeformat)) %>%
  group_by(month) %>%
  summarise(rating_avg = mean(rating)) %>%
  ggplot(aes(month, rating_avg)) +
  geom_point() +
  labs(title = "General distribution of the ratings",
       x = "Month in Numbers",
       y = "Avg. Rating")

######## Display the prevelance of ratings based on ratings and prevelance. ########

edx %>%
  mutate(month = month(timeformat)) %>%
  group_by(month) %>%
  summarise(count = n()) %>%
  ggplot(aes(month, count/1000000)) +
  geom_point() +
  labs(title = "Prevelance of ratings in diffrent months",
       x = "Month in Numbers",
       y = "Prevelance of Ratings in millions")

####### Visualize the general distribution of the ratings ###########

edx %>%
  group_by(rating) %>%
  summarise(count = n()) %>%
  ggplot(aes(rating, count/1000000)) +
  geom_col() +
  labs(title = "General distribution of the ratings",
       x = "Rating",
       y = "Prevelance in millions")

############## Visualize the most rated movies in descending order ###############

edx %>%
  group_by(title) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(10) %>%
  ggplot(aes(reorder(title, -count), count)) +
  geom_col() +
  scale_x_discrete(guide = ggplot2::guide_axis(n.dodge = 2), 
                   labels = function(x) stringr::str_wrap(x, width = 20)) +
  labs(title = "Ten most rated movies",
       x = "Movie",
       y = "Prevelance")


######################## ANALYTICS ##########################################

###### Split EDX into a training set (80%) and test set (20%).

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp_test_set <- edx[test_index,]

# Make sure userId and movieId in test set are also in training set
test_set <- temp_test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into training set
removed <- anti_join(temp_test_set, test_set)
train_set <- rbind(train_set, removed)

# Using the naive model --> Calculate the mean based on ratings to use for predictions.

mu <- mean(train_set$rating)

# predict the ratings given only the mean(rating).
predicted_ratings <- test_set %>%
  mutate(pred = mu)

# calculate and display the RMSE for mean(rating).
model_0_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- data_frame(method = "Mean(Rating)", RMSE = model_0_rmse)
rmse_results %>% knitr::kable()

##################################### Checking for Correlation and Improving Prediciton Model ##########################################

# Calculate the influence of movie to the rating.
movie_vals <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i_ind = mean(rating - mu))

# Calculate the influence of user to the rating.
user_vals <- train_set %>%
  group_by(userId) %>% 
  summarize(b_u_ind = mean(rating - mu))

# Calculate the influence of genre to the rating.
genre_vals <- train_set %>%
  group_by(genres) %>% 
  summarize(b_g_ind = mean(rating - mu))

# Calculate the influence of monthyear to the rating.
monthyear_vals <- train_set %>%
  group_by(monthyear) %>% 
  summarize(b_my_ind = mean(rating - mu))

#Correlation between rating and the movie predictor
cor_rating_movie_ind <- test_set %>% 
  left_join(movie_vals, by='movieId') %>%
  summarize(r = cor(rating, b_i_ind))

cor_results <- data_frame(Variables = "Rating ~ Movie (Independent)", Correlation = cor_rating_movie_ind$r)

#Correlation between rating and the user predictor
cor_rating_user_ind <- test_set %>% 
  left_join(user_vals, by='userId') %>%
  summarize(r = cor(rating, b_u_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ User (Independent)", Correlation = cor_rating_user_ind$r))

#Correlation between rating and the genre predictor
cor_rating_genre_ind <- test_set %>% 
  left_join(genre_vals, by='genres') %>%
  summarize(r = cor(rating, b_g_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ Genre (Independent)", Correlation = cor_rating_genre_ind$r))

#Correlation between rating and the monthyear predictor
cor_rating_monthyear_ind <- test_set %>% 
  left_join(monthyear_vals, by='monthyear') %>%
  summarize(r = cor(rating, b_my_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ Monthyear (Independent)", Correlation = cor_rating_monthyear_ind$r))
cor_results %>% knitr::kable()

# Calculate the influence of movies to the rating.
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Calculate the influence the predictor has on the model.
model_1_predictor <- sqrt(mean((movie_avgs$b_i)^2))

# predict the ratings given the movie effect.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i)

# calculate and display the RMSE for mean(rating) + movie.
model_1_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Movie",  RMSE = model_1_rmse, Predictorweight = model_1_predictor))
rmse_results %>% knitr::kable()

#Correlation between movie predictor and the user predictor
cor_movie_user_ind <- test_set %>% 
  left_join(movie_vals, by='movieId') %>%
  left_join(user_vals, by='userId') %>%
  summarize(r = cor(b_i_ind, b_u_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Movie ~ User (Independent)", Correlation = cor_movie_user_ind$r))
cor_results %>% knitr::kable()

# Calculate the influence of users to the rating given the movie effect.
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

# Calculate the influence the predictor has on the model.
model_2_predictor <- sqrt(mean((user_avgs$b_u)^2))

# predict the ratings given the movie + user effect.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u)

# calculate and display the RMSE for mean(rating) + movie + user.
model_2_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Movie + User",  RMSE = model_2_rmse, Predictorweight = model_2_predictor))
rmse_results %>% knitr::kable()

#Correlation between rating and the user predictor depending on the movie predictor
cor_movie_user <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  summarize(r = cor(rating, b_u))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ User (Movie)", Correlation = cor_movie_user$r))
cor_results %>% knitr::kable()

#Correlation between movie predictor and the genre predictor
cor_movie_genre_ind <- test_set %>% 
  left_join(movie_vals, by='movieId') %>%
  left_join(genre_vals, by='genres') %>%
  summarize(r = cor(b_i_ind, b_g_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Movie ~ Genre (Independent)", Correlation = cor_movie_genre_ind$r))

#Correlation between user predictor and the genre predictor
cor_user_genre_ind <- test_set %>% 
  left_join(user_vals, by='userId') %>%
  left_join(genre_vals, by='genres') %>%
  summarize(r = cor(b_u_ind, b_g_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "User ~ Genre (Independent)", Correlation = cor_user_genre_ind$r))
cor_results %>% knitr::kable()

# Calculate the influence of the genre to the rating given the movie + user effect.
genre_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Calculate the influence the predictor has on the model
model_3_predictor <- sqrt(mean((genre_avgs$b_g)^2))

# predict the ratings given the movie + user + genre effect.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g)

# calculate and display the RMSE for mean(rating) + movie + user + genre.
model_3_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Movie + User + Genres",  RMSE = model_3_rmse, Predictorweight = model_3_predictor ))
rmse_results %>% knitr::kable()

#Correlation between rating and the genre predictor depending on the movie + user predictor
cor_user_genre <- test_set %>% 
  left_join(genre_avgs, by='genres') %>%
  summarize(r = cor(rating, b_g))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ Genre (Movie + User)", Correlation = cor_user_genre$r))
cor_results %>% knitr::kable()

#Correlation between movie predictor and the monthyear predictor
cor_movie_monthyear_ind <- test_set %>% 
  left_join(movie_vals, by='movieId') %>%
  left_join(monthyear_vals, by='monthyear') %>%
  summarize(r = cor(b_i_ind, b_my_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Movie ~ Monthyear (Independent)", Correlation = cor_movie_monthyear_ind$r))

#Correlation between user predictor and the monthyear predictor
cor_user_monthyear_ind <- test_set %>% 
  left_join(user_vals, by='userId') %>%
  left_join(monthyear_vals, by='monthyear') %>%
  summarize(r = cor(b_u_ind, b_my_ind))

cor_results <- bind_rows(cor_results, data_frame(Variables = "User ~ Monthyear (Independent)", Correlation = cor_user_monthyear_ind$r))
cor_results %>% knitr::kable()

# Calculate the influence of the monthyear to the rating given the movie + user effect
monthyear_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(monthyear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u))

# Calculate the influence the predictor has on the model
model_4_predictor <- sqrt(mean((monthyear_avgs$b_my)^2))

# predict the ratings given the movie + user + monthyear effect.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(monthyear_avgs, by='monthyear') %>%
  mutate(pred = mu + b_i + b_u + b_my)

# calculate and display the RMSE for mean(rating) + movie + user + monthyear.
model_4_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Movie + User + MonthYear",  RMSE = model_4_rmse, Predictorweight = model_4_predictor ))
rmse_results %>% knitr::kable()

#Correlation between rating and the monthyear predictor depending on the movie + user predictor
cor_user_monthyear <- test_set %>% 
  left_join(monthyear_avgs, by='monthyear') %>%
  summarize(r = cor(rating, b_my))

cor_results <- bind_rows(cor_results, data_frame(Variables = "Rating ~ Monthyear (Movie + User)", Correlation = cor_user_monthyear$r))
cor_results %>% knitr::kable()


########## Regularization to find the optimal ammount of ratings #####

# Using crossvalidation to get the optimal lambda for the movie effect.
lambdas <- seq(0, 10, 0.5)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred = mu + b_i)
  return(sqrt(mean((test_set$rating - predicted_ratings$pred)^2)))
})

# plot the different lambdas against the RMSEs to select lambda with the lowest RMSE value.
qplot(lambdas, rmses)  

# Get the lambda with the lowest RMSE.
lambda_movie <- lambdas[which.min(rmses)]

# Using crossvalidation to get the optimal lambda for the user effect.
lambdas <- seq(0, 10, 0.5)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda_movie))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u)
  return(sqrt(mean((test_set$rating - predicted_ratings$pred)^2)))
})

# plot the different lambdas against the RMSEs to select lambda with the lowest RMSE value.
qplot(lambdas, rmses)  

# Get the lambda with the lowest RMSE.
lambda_user <- lambdas[which.min(rmses)]

#Calculating the movie effect using the corresponding lambda
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda_movie))

# Calculate the influence the predictor has on the model
model_5_predictor <- sqrt(mean((b_i$b_i)^2))

# predict the ratings given the generalized movie effect.
predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i)

# calculate and display the RMSE for mean(rating) + generalized movie.
model_5_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Generalized Movie",  RMSE = model_5_rmse, Predictorweight = model_5_predictor ))
rmse_results %>% knitr::kable()

#Calculating the user effect using the corresponding lambda of user.
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_user))

# Calculate the influence the predictor has on the model
model_6_predictor <- sqrt(mean((b_u$b_u)^2))

# predict the ratings given the generalized movie and generalized user effect.
predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

# calculate and display the RMSE for mean(rating) + generalized movie and generalized user.
model_6_rmse <- sqrt(mean((test_set$rating - predicted_ratings$pred)^2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Generalized Movie + User",  RMSE = model_6_rmse, Predictorweight = model_6_predictor ))
rmse_results %>% knitr::kable()


#Final Model: Generalized Movie + User.

#training the model with the mean of the training set edx
mu <- mean(edx$rating)

#including the movie-effect based on the training set edx and the corresponding lambda
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda_movie))

#including the user-effect based on the training set edx and the corresponding lambda
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_user))

#predicting the ratings based on the generalized model using the predictors movie and user
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)
model_7_rmse <- sqrt(mean((validation$rating - predicted_ratings$pred)^2))
rmse_results <- data_frame(method = "Generalized Movie + User", RMSE = model_7_rmse)
rmse_results %>% knitr::kable()