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
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
library(plyr)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(e1071)
library(klaR)
```

```
## Loading required package: MASS
```

```r
library(gbm)
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
```

```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
## 
## Attaching package: 'pROC'
## 
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
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
We split the data into training/test sets with a 60/40 split. The `filename` column is omitted as this contains no information. We also remove parameters that were saved from the method used to produce the data samples.

```r
split <- createDataPartition(data$trigger_index,p=0.6,list=FALSE)
colRemove <- c(1,grep("global|type",names(data)))
training <- data[split,-colRemove]
testing <- data[-split,-colRemove]
```

## Exploratory Plots
Our first plot shows the distribution of log-Luminosity (`log_L`) vs redshift (`z`) for both triggered and missed GRBs.

```r
qplot(training$log_L,training$z,color=training$trigger_index,xlab="log(Luminosity)",ylab="redshift",main="Scatterplot of luminosity and redshift for GRBs")
```

![plot of chunk expl_plot_1](./swift_pipe_files/figure-html/expl_plot_1.png) 
