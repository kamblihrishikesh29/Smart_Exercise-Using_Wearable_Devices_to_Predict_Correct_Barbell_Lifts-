---
title: "Course_Project_PML"
author: "Hrishikesh Kambli"
date: "2/15/2021"
output: html_document
---



# I. Overview
This document is the final report of the Peer Assessment project from Coursera's course Practical Machine Learning.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. The data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants was recorded as they were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The main goal of the project is to predict the manner in which 6 participants performed these exercise as described below. This is the "classe" variable in the training set. The machine learning algorithm described here is applied to the 20 test cases available in the test data.

# II. Data Loading and Exploratory Analysis

### a) Dataset Overview
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

A short description of the datasets content from the authors website:

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)."

### b) Loading required libraries

```r
library(caret)
library(rattle)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)
set.seed(12345)
```


### c) Data Loading and Cleaning
The next step is loading the dataset from the URL provided above.


```r
train <- read.csv("C:/Users/kambl/Documents/R/Practical Machine Learning/pml-training.csv")
test <- read.csv("C:/Users/kambl/Documents/R/Practical Machine Learning/pml-testing.csv")
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing dataset is not changed.

```r
inTrain  <- createDataPartition(train$classe, p=0.7, list=FALSE)
TrainSet <- train[inTrain, ]
ValidSet  <- train[-inTrain, ]
dim(TrainSet)
```

```
## [1] 13737   160
```


```r
dim(ValidSet)
```

```
## [1] 5885  160
```

Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. 

Removing variables with Nearly Zero Variance.

```r
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
ValidSet  <- ValidSet[, -NZV]
dim(TrainSet)
```

```
## [1] 13737   104
```


```r
dim(ValidSet)
```

```
## [1] 5885  104
```


Removing the variables that contains missing values.

```r
TrainSet<- TrainSet[, colSums(is.na(TrainSet)) == 0]
ValidSet <- ValidSet[, colSums(is.na(ValidSet)) == 0]
dim(TrainSet)
```

```
## [1] 13737    59
```


```r
dim(ValidSet)
```

```
## [1] 5885   59
```

Removing identification only variables that don't make intuitive sense for prediction

```r
TrainSet <- TrainSet[, -(1:5)]
ValidSet  <- ValidSet[, -(1:5)]
dim(TrainSet)
```

```
## [1] 13737    54
```


```r
dim(ValidSet)
```

```
## [1] 5885   54
```

With the cleaning process above, the number of variables for the analysis has been reduced to 54 only.


# III. Prediction Model Building
Three methods will be applied to model the regressions (in the Train dataset).
The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below.

### a) Method: Random Forest


```r
# model fit
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRF <- train(classe ~ ., data=TrainSet, method="rf",
                  trControl=controlRF)
modFitRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B    6 2647    4    1    0 0.0041384500
## C    0    5 2391    0    0 0.0020868114
## D    0    0    9 2243    0 0.0039964476
## E    0    0    0    5 2520 0.0019801980
```



```r
# prediction on Validation dataset
predictRF <- predict(modFitRF, newdata=ValidSet)
confMatRF <- confusionMatrix(predictRF, as.factor(ValidSet$classe))
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    2    0    0
##          C    0    0 1024    2    0
##          D    0    0    0  962    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9978, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   0.9981   0.9979   0.9991
## Specificity            0.9998   0.9996   0.9996   0.9998   1.0000
## Pos Pred Value         0.9994   0.9982   0.9981   0.9990   1.0000
## Neg Pred Value         1.0000   0.9998   0.9996   0.9996   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1934   0.1740   0.1635   0.1837
## Detection Prevalence   0.2846   0.1937   0.1743   0.1636   0.1837
## Balanced Accuracy      0.9999   0.9994   0.9988   0.9989   0.9995
```

The accuracy is 99.9%, thus my predicted accuracy for the out-of-sample error is 0.1%.

### b) Method: Decision Trees


```r
# model fit
set.seed(12345)
modFitDT <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDT)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

```r
prp(modFitDT)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)



```r
# prediction on Validation dataset
predictDT <- predict(modFitDT, newdata= ValidSet, type="class")
confMatDT <- confusionMatrix(predictDT, as.factor(ValidSet$classe))
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1502  201   59   66   74
##          B   58  660   37   64  114
##          C    4   66  815  129   72
##          D   90  148   54  648  126
##          E   20   64   61   57  696
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7342          
##                  95% CI : (0.7228, 0.7455)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6625          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8973   0.5795   0.7943   0.6722   0.6433
## Specificity            0.9050   0.9425   0.9442   0.9151   0.9579
## Pos Pred Value         0.7897   0.7074   0.7505   0.6079   0.7751
## Neg Pred Value         0.9568   0.9033   0.9560   0.9344   0.9226
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2552   0.1121   0.1385   0.1101   0.1183
## Detection Prevalence   0.3232   0.1585   0.1845   0.1811   0.1526
## Balanced Accuracy      0.9011   0.7610   0.8693   0.7936   0.8006
```
The accuracy is 73.42%.

### c) Method: Generalized Boosted Model

```r
# model fit
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```



```r
# prediction on Validation dataset
predictGBM <- predict(modFitGBM, newdata=ValidSet)
confMatGBM <- confusionMatrix(predictGBM, as.factor(ValidSet$classe))
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   12    0    1    0
##          B    6 1115   12    1    3
##          C    0   12 1012   21    0
##          D    0    0    2  941    6
##          E    0    0    0    0 1073
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9871          
##                  95% CI : (0.9839, 0.9898)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9837          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9789   0.9864   0.9761   0.9917
## Specificity            0.9969   0.9954   0.9932   0.9984   1.0000
## Pos Pred Value         0.9923   0.9807   0.9684   0.9916   1.0000
## Neg Pred Value         0.9986   0.9949   0.9971   0.9953   0.9981
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1895   0.1720   0.1599   0.1823
## Detection Prevalence   0.2856   0.1932   0.1776   0.1613   0.1823
## Balanced Accuracy      0.9967   0.9871   0.9898   0.9873   0.9958
```

# IV. Applying the Selected Model to the Test Data

The accuracy of the 3 regression modeling methods above are:
  Random Forest : 0.999
Decision Tree : 0.7342
GBM : 0.987
In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.


```r
predictTEST <- predict(modFitRF, newdata=test)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

