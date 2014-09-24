calculatePredictions <- function(pthresh, true, pred) {
        p <- sum(true==1)
        n <- sum(true==0)
        tp <- sum(true==1 & pred>=pthresh)
        fn <- sum(true==1 & pred<pthresh)
        fp <- sum(true==0 & pred>=pthresh)
        tn <- sum(true==0 & pred<pthresh)
        tpr <- tp/p
        fpr <- fp/n
        acc <- (tp+tn)/(p+n)
        f1 <- 2*tp/(2*tp+fp+fn)
        c(pthresh,tp,fn,fp,tn,tpr,fpr,acc,f1)
}

calculateROC <- function(true, pred, d = 0.01) {
        p <- seq(d,1-d,by=d)
        roc <- sapply(p,calculatePredictions,true = true, pred = pred)
        roc <- data.frame(t(roc))
        colnames(roc) <- c("pthresh","TP","FN","FP","TN","TPR","FPR","Accuracy","F1score")
        roc
}

plotROC <- function(roc,title="") {
        idx <-  which.max(roc$F1score)
        plot(log10(roc$FPR),roc$TPR,type="l",xlab="log10(False Positive Rate)",ylab="True Positive Rate",main=title)
        points(log10(roc$FPR[idx]),roc$TPR[idx],pch=3,cex=3,type="p")
}