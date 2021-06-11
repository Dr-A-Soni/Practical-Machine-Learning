

## Loading the libraries of hte required packages:
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
require(dplyr)
library(reshape2)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(randomForest)
library(RColorBrewer)


## Loading and cleaning the data:


testing_data <- read.csv("pml-testing.csv",strip.white = TRUE, na.strings=c('#DIV/0!', '', 'NA'))
training_data <- read.csv("pml-training.csv",strip.white = TRUE ,na.strings=c('#DIV/0!', '', 'NA'))

dim(testing_data)
dim(training_data)


## Partitioning the training data set into training set and validation set:
Intrain <- createDataPartition(training_data$classe, p = 0.75, list = FALSE)
train <- training_data[Intrain, ]
valid <- training_data[-Intrain, ]

dim(Intrain)
dim(train)
dim(valid)


# Now clean-up the variables w/ zero variance
# Be careful, kick out the same variables in both cases

##Now removing the data which are irrelevant for the prediction:
# cleaning the data with zero variances in the variables:


nzv <- nearZeroVar(train)
nzv
train <- train[,-nzv]
valid <- valid[,-nzv]


dim(train)
dim(valid)

## there seems to be variables with NA, hence cleaning and sorting the data 
#which are useful for the prediction model:

fval <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[,fval == FALSE]

## Following the same step on the validation dataset:
#sval <- sapply(valid, function(x) mean(is.na(x))) > 0.95
valid <- valid[,fval == FALSE]
## the roughly gathered dataset:

dim(train)
dim(valid)

## Now both the train and valid dataset contain variables that are resuced to 59 
## which  previously had 160. 
## These 59 variables in the dataset are useful in further prediction models and 
## determing the answer, However the first five variables are the identifiers and 
## does not play any role in the predictions hence the best option would be to 
## remove them form the data set.

train <- train[, -c(1:5)]
valid <- valid[, -c(1:5)]

dim(train)
dim(valid)

## Now the data is clean and can be used for building the prediction models.


## Before building the models lets try to do the correlation analysis:

#Select the first FPC for the first pricipal component order:

correlation_matrix <- cor(train[ , -54])
corrplot(correlation_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0,0,0))
  
## We will try to build models:
# since i am trying to learn more I wold like to try out 
# decision tree, a random forest,  a generalized boosted model and SVM:

## Not that we will use the train dataset for these model which are filtered
  #  from unwanted variables, than we will test them on the validation set under
  ## the dataset valid. 

set.seed(1234)
## Prediction Models:

## we will try out all model to find best model for testing the final testing dataset.

#1) decision Tree:

Dee_Tree <- train(classe ~., data = train, method = "rpart",
                  trControl = trainControl(method = "cv", number = 3), 
                  tuneLength = 5)

fancyRpartPlot(Dee_Tree$finalModel)

predict_DEE_TREE <- predict(Dee_Tree, newdata = valid)

y <- as.factor(valid$classe)
DT_CM <- confusionMatrix(predict_DEE_TREE, y)
print(DT_CM)

# 2) random forest:
set.seed(1234)

Forest_model <- train(classe ~., data = train, method = "rf",
                      trControl = trainControl(method = "repeatedcv",
                                               number = 5,repeats =2), 
                      verbose = FALSE)
Forest_model$finalModel

#The Prediction of the Random Forest Model is now done on the valid dataset.

predic_FM <- predict(Forest_model, newdata = valid)
predic_FM
x <- as.factor(valid$classe)
cM_FM <- confusionMatrix(predic_FM, x)
print(cM_FM)



# 3) Generalized boosted Model:
set.seed(1234)
GBM_model <- train(classe ~., data = train, method = "gbm",
                  trControl = trainControl(method = "repeatedcv", number = 5,
                                           repeats = 1), verbose = FALSE)

# Now lets use this Model to predict on the validation set which we separated previously

Predict_GBM <- predict(GBM_model, newdata = valid)

z <- as.factor(valid$classe)

GBM_CM <- confusionMatrix(Predict_GBM, z)
print(GBM_CM)



# 4) Svm :

SVM_model <- train(classe ~., data = train, method = "svmLinear",
                  trControl = trainControl(method = "cv", number = 3), 
                  tuneLength = 5)

SVM_model$finalModel
  
  
# Prediction time on the validation data set:

Predict_SVM <- predict(SVM_model, newdata = valid)
V <- as.factor(valid$classe)
SVM_CM <- confusionMatrix(Predict_SVM, V)
print(SVM_CM)

# Now lets see the result of all the accuracy of all the models
#To summarize the accuracy of all the  models:
#lets collect the Accuracy form all the Models and find which one
#is the most accurate one

tab <- as.matrix(c(DT_CM$overall[1],cM_FM$overall[1],GBM_CM$overall[1],SVM_CM$overall[1]))
rownames(tab) <- c("Decision Tree", "Random Forest", "GBM", "SVM")
colnames(tab) <- ("Accuracy")
tab

#Looking at the data the best models are Random Forest and GBM with 0.9977% and 
# 0.9875% accuracy.
# We will use the Random Model on the final testing_data set aside initially 
# to predict.


##Applying the best Predictive Model to the Test set 

Predict_final <- predict(Forest_model, newdata = testing_data)
Predict_final
