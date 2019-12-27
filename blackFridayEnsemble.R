library(dplyr)
library(purrr)
library(magrittr)
library(caret)
library(Matrix)
library(glmnet)
library(ggplot2)
library(forcats)
library(randomForest)
library(Matrix)
library(xgboost)
library(ModelMetrics)

df_raw <- read.csv('BlackFriday.csv')
str(df_raw)
df_raw %>% View

#Preprocessing
df <- df_raw %>%  mutate(Product_Category_2 = as.factor(ifelse(is.na(Product_Category_2), 'NA', as.character(Product_Category_2))),
                         Product_Category_3 = as.factor(ifelse(is.na(Product_Category_3), 'NA', as.character(Product_Category_3))),
                         User_ID = as.factor(User_ID),
                         Product_ID = as.factor(Product_ID),
                         Occupation = as.factor(Occupation),
                         Marital_Status = as.factor(Marital_Status),
                         Product_Category_1 = as.factor(Product_Category_1))

#df$User_ID <- NULL
#df$Product_ID <- NULL

#FeatureEngineering
#df %<>%
#  mutate(genderAge = paste0(Gender, Age),
#         occupationMarital = paste0(Occupation, Marital_Status),
#         categories = paste0(Product_Category_1, Product_Category_2, Product_Category_3),
#         home = paste0(City_Category, Stay_In_Current_City_Years))

#data partitioning
splitIndices <- createDataPartition(df$Purchase, p = 0.75, list = FALSE, times = 1)
df_train <- df[splitIndices[,1],]
df_test <- df[-splitIndices[,1],]

#EDA. Alot of potential for deep meaningful insights. 
df %>% 
  ggplot(aes(x = Gender, y = Purchase)) + 
  geom_boxplot()


df %>% select(User_ID, Purchase) %>% 
  group_by(User_ID) %>% 
  summarise(totalSpent = sum(Purchase),
            Items = n())

df %>% group_by(Occupation) %>% 
  summarise(totalSpent = mean(Purchase)) %>% 
  ggplot(aes(x = totalSpent, y = fct_reorder(Occupation, totalSpent))) + geom_point() +
  ylab('Occupation')

#Feature Scaling
#df_train$Purchase <- scale(df_train$Purchase)
#df_test$Purchase <- scale(df_test$Purchase)


#XGBoost
#fullData <- rbind(df_train, df_test)
sprsMat <- sparse.model.matrix(Purchase ~ ., data = df)
trnMat <- sprsMat[splitIndices[,1],]
tstMat <- sprsMat[-splitIndices[,1],]
#trainm <- sparse.model.matrix(Purchase ~ ., data = df_train)

train_matrix <- xgb.DMatrix(trnMat, label = df_train$Purchase)

#testm <- sparse.model.matrix(Purchase ~ ., data = df_test)
test_matrix <- xgb.DMatrix(tstMat, label = df_test$Purchase)

watchList <- list(train = train_matrix, test = test_matrix)

 model_fit <- xgb.train(data = train_matrix,
                        object = 'reg:linear',
                        nrounds = 300,
                        booster = 'gbtree',
                        max_depth = 16,
                        subsample = 0.5,
                        min_child_weight = 0.8,
                        eta = 0.4,
                        watchlist = watchList)
 
 model_fit1 <- xgb.train(data = train_matrix,
                         object = 'reg:linear',
                         nrounds = 250,
                         booster = 'gbtree',
                         max_depth = 30,
                         subsample = 0.7,
                         eta = 0.4)
 
 model_fit2 <-  xgb.train(data = train_matrix,
                          object = 'reg:linear',
                          nrounds = 250,
                          booster = 'gbtree',
                          max_depth = 20,
                          subsample = 0.5,
                          eta = 0.4)
 
 model_fit3 <- xgb.train(data = train_matrix,
                         object = 'reg:linear',
                         nrounds = 250,
                         booster = 'gbtree',
                         max_depth = 40,
                         subsample = 0.5,
                         eta = 0.05)
 
 
 

# CrossValidate
# params = list(objective = 'reg:linear',
#               eta = 0.1,
#               max_depth = 12,
#               subsample = 0.5,
#               min_child_weight = 0.8,
#               booster = 'gbtree')
# modelCV <- xgb.cv(params = params, data = train_matrix, nrounds = 300, nfold = 5,
#        showsd=T, early_stopping_rounds = 20)
# 
# modelCV$best_iteration
# 
# model_fit

#error plot - train vs test
e <- data.frame(model_fit$evaluation_log)
plot(e$iter,e$train_rmse, col = 'blue')
lines(e$iter, e$test_rmse, col = 'red')


model_fit$feature_names
model_fit$nfeatures

#feature importance
imp <- xgb.importance(feature_names = model_fit$feature_names,
               model = model_fit,
               data = df_train)

imp %>% View
xgb.plot.importance(imp)

pred <- predict(model_fit, test_matrix)
pred1 <- predict(model_fit1, test_matrix)
pred2 <- predict(model_fit2, test_matrix)
pred3 <- predict(model_fit3, test_matrix)
rmse(df_test$Purchase, (pred+pred2+pred1+pred3)/4)

dim(test_matrix)
dim(train_matrix)


#GLMNET
dataMatrix <- sparse.model.matrix(Purchase ~ ., df_train)

model_fit1 <- glmnet(x = dataMatrix,
                    y = df_train$Purchase,
                    family = 'gaussian',
                    alpha =0.1,
                    standardize = FALSE)

pred_matrix <- sparse.model.matrix(Purchase ~ ., df_test)

prediction <- predict.glmnet(model_fit1, pred_matrix, type = 'response')


0.7*pred+0.3*prediction
rmse(df_test$Purchase, 0.95*pred+0.05*prediction)

dim(pred_matrix)
dim(dataMatrix)


head(df_train)
head(df_test)


