# Swift GRB Pipeline Modeling
Philip Graff  
September 23, 2014  


# Introduction

The Swift satellite's detection pipeline uses over 500 triggering criteria to detect long gamma-ray bursts. Modeling of the GRB and this pipeline is extremely computationally expensive, so here we aim to replace the original model with an approximation trained via machine learning.

# Loading Libraries

```r
library(lattice)
library(ggplot2)
library(caret)
library(rattle)
library(rpart)
library(plyr)
library(randomForest)
library(e1071)
library(klaR)
library(gbm)
library(ada)
library(pROC)
source("ROCfunctions.R")
```

# Loading the Data
Let's begin our analysis with the newest set of prior samples. The column `trigger_index` indicates if a GRB was detected (1) or not (0).

```r
data <- read.table("data/test_sample_prior2.txt",header=TRUE)
names(data)
```

```
##  [1] "filename"          "log_L"             "z"                
##  [4] "grid_id"           "bin_size_emit"     "alpha"            
##  [7] "beta"              "E_peak"            "background_name"  
## [10] "bgd_15.25keV"      "bgd_15.50keV"      "bgd25.100keV"     
## [13] "bgd50.350keV"      "theta"             "flux"             
## [16] "burst_shape"       "ndet"              "rate_GRB_0_global"
## [19] "z1_global"         "n1_global"         "n2_global"        
## [22] "lum_star_global"   "alpha1_global"     "beta1_global"     
## [25] "Epeak_type"        "Lum_evo_type"      "trigger_index"
```

```r
dim(data)
```

```
## [1] 10000    27
```

```r
data$trigger_index <- as.factor(data$trigger_index)
```

## Split into Train/Test
We split the data into training/test sets with a 60/40 split. The `filename` column is omitted as this contains no information. We also remove parameters that were saved from the method used to produce the data samples and the names of the backgrounds and burst shapes used for each sample.

```r
split <- createDataPartition(data$trigger_index,p=0.6,list=FALSE)
colRemove <- c(1,grep("global|type|background|burst",names(data)))
training <- data[split,-colRemove]
testing <- data[-split,-colRemove]
```

## Exploratory Plots
Our first plot shows the distribution of log-Luminosity (`log_L`) vs redshift (`z`) for both triggered and missed GRBs.


```r
qplot(training$log_L,training$z,color=training$trigger_index,xlab="log(Luminosity)",ylab="redshift",main="Scatterplot of Luminosity and Redshift for GRBs")
```

![plot of chunk expl_plot_1](figure/expl_plot_1.png) 

We also look at the distribution of redshifts of detected GRBs.


```r
qplot(training$z[training$trigger_index==1],xlab="redshift",ylab="frequency",main="Histogram of Redshift for Detected GRBs")
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![plot of chunk expl_plot_2](figure/expl_plot_2.png) 

# Machine Learning Analysis
We now train multiple machine learning methods on the problem. Each is then evaluated on the test data for comparison and analyzed using a ROC.

## Random Forests
Random forests uses an ensemble of decision trees. The default of 500 trees per forest is kept.

```r
modRF <- train(trigger_index ~ ., data = training, method = "rf", tuneGrid = expand.grid(mtry = seq(3,13,by=2)), trControl = trainControl(method = "cv", number = 10))
modRF
```

```
## Random Forest 
## 
## 6001 samples
##   14 predictor
##    2 classes: '0', '1' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 5401, 5401, 5400, 5402, 5400, 5402, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    3    1         1      0.005        0.012   
##    5    1         1      0.005        0.013   
##    7    1         1      0.003        0.008   
##    9    1         1      0.003        0.008   
##   11    1         1      0.004        0.010   
##   13    1         1      0.003        0.008   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 11.
```

This is now evaluated on the test data set.

```r
predRF1 <- predict(modRF$finalModel, testing, type="response")
predRF2 <- predict(modRF$finalModel, testing, type="prob")
confMatRF <- confusionMatrix(predRF1, testing$trigger_index)
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 2868   21
##          1   17 1093
##                                         
##                Accuracy : 0.99          
##                  95% CI : (0.987, 0.993)
##     No Information Rate : 0.721         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.976         
##  Mcnemar's Test P-Value : 0.626         
##                                         
##             Sensitivity : 0.994         
##             Specificity : 0.981         
##          Pos Pred Value : 0.993         
##          Neg Pred Value : 0.985         
##              Prevalence : 0.721         
##          Detection Rate : 0.717         
##    Detection Prevalence : 0.722         
##       Balanced Accuracy : 0.988         
##                                         
##        'Positive' Class : 0             
## 
```

We plot a ROC to see performance varying over threshold value.

```r
rocRF <- calculateROC(testing$trigger_index, predRF2[,2], d = 0.001)
plotROC(rocRF, title = "Random Forests ROC")
```

![plot of chunk rf_roc](figure/rf_roc.png) 

The optimal threshold is at a probability of 0.435 and achieves an accuracy of 99.1998%.

## AdaBoost
AdaBoost performs boosting of decision trees, where an ensemble of trees is used. Each time a tree is trained, samples predicted correctly are down-weighted and samples predicted incorrectly are up-weighted. We tune over the number of trees, the tree depth, and learning rate.

```r
modAda <- train(trigger_index ~ ., data = training, method = "ada", tuneGrid = expand.grid(nu=0.1, iter=50, maxdepth=3), trControl = trainControl(method = "cv"))
modAda
```

```
## Boosted Classification Trees 
## 
## 6001 samples
##   14 predictor
##    2 classes: '0', '1' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 5401, 5401, 5401, 5400, 5402, 5401, ... 
## 
## Resampling results
## 
##   Accuracy  Kappa  Accuracy SD  Kappa SD
##   1         1      0.004        0.009   
## 
## Tuning parameter 'iter' was held constant at a value of 50
## 
## Tuning parameter 'maxdepth' was held constant at a value of 3
## 
## Tuning parameter 'nu' was held constant at a value of 0.1
## 
```

This is now evaluated on the test data set.

```r
predAda1 <- predict(modAda$finalModel, testing, type="vector")
predAda2 <- predict(modAda$finalModel, testing, type="prob")
confMatAda <- confusionMatrix(predAda1, testing$trigger_index)
confMatAda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 2866   21
##          1   19 1093
##                                         
##                Accuracy : 0.99          
##                  95% CI : (0.986, 0.993)
##     No Information Rate : 0.721         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.975         
##  Mcnemar's Test P-Value : 0.874         
##                                         
##             Sensitivity : 0.993         
##             Specificity : 0.981         
##          Pos Pred Value : 0.993         
##          Neg Pred Value : 0.983         
##              Prevalence : 0.721         
##          Detection Rate : 0.717         
##    Detection Prevalence : 0.722         
##       Balanced Accuracy : 0.987         
##                                         
##        'Positive' Class : 0             
## 
```

We plot a ROC to see performance varying over threshold value.

```r
rocAda <- calculateROC(testing$trigger_index, predAda2[,2], d = 0.001)
plotROC(rocAda, title = "AdaBoost ROC")
```

![plot of chunk ada_roc](figure/ada_roc.png) 

The optimal threshold is at a probability of 0.673 and achieves an accuracy of 99.0998%.

## Support Vector Machines
Support vector machines...

```r
modSVM <- train(trigger_index ~ ., data = training, method = "svmRadial", tuneLength = 3, preProc = c("center", "scale"), trControl = trainControl(method = "cv", classProbs = TRUE))
modSVM
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 6001 samples
##   14 predictor
##    2 classes: '0', '1' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 5400, 5402, 5401, 5401, 5402, 5401, ... 
## 
## Resampling results across tuning parameters:
## 
##   C    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.2  0.9       0.9    0.008        0.02    
##   0.5  1.0       0.9    0.009        0.02    
##   1.0  1.0       0.9    0.009        0.02    
## 
## Tuning parameter 'sigma' was held constant at a value of 0.07452
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.07452 and C = 1.
```

This is now evaluated on the test data set.

```r
#predSVM1 <- predict(modSVM$finalModel, testing, type="response")
#predSVM2 <- predict(modSVM$finalModel, testing, type="prob")
#confMatSVM <- confusionMatrix(predSVM1, testing$trigger_index)
#confMatSVM
```

We plot a ROC to see performance varying over threshold value.

```r
#rocSVM <- calculateROC(testing$trigger_index, predSVM2[,2], d = 0.001)
#plotROC(rocSVM, title = "Support Vector Machines ROC")
```


## Linear/Quadratic Discriminant Analysis

## Model Stacking
