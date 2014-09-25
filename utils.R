gridToXY <- function(grid) {
        x <- (grid %% 7 - 3) / 2
        y <- (as.integer(grid/7) - 2) / 3
        r <- sqrt(x^2 + y^2)
        phi <- atan2(y,x)
        c(r,phi)
}

mySVM <- list(type = "Classification", library = "e1071", loop = NULL)
prm <- data.frame(parameter = c("cost", "gamma"),
                  class = rep("numeric", 2),
                  label = c("Cost", "Gamma"))
mySVM$parameters <- prm
svmGrid <- function(x, y, len = NULL) {
        library(kernlab)
        ## This produces low, middle and high values for sigma 
        ## (i.e. a vector with 3 elements). 
        sigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
        expand.grid(gamma = mean(sigmas[-2]),
                    cost = 2 ^((1:len) - 3))
}
mySVM$grid <- svmGrid
svmFit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
        svm(x = as.matrix(x), y = y,
            kernel = "radial",
            gamma = param$gamma,
            cost = param$cost,
            probability = classProbs,
            ...)
}
mySVM$fit <- svmFit
svmPred <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
        predict(modelFit, newdata)
mySVM$predict <- svmPred
svmProb <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
        predict.svm(modelFit, newdata, probability=TRUE)
mySVM$prob <- svmProb
svmSort <- function(x) x[order(x$cost), ]
mySVM$sort <- svmSort
mySVM$levels <- function(x) lev(x)
