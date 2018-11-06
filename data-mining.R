
setwd("")

## function that implements random forest + 10 fold cross validation
## the response of the data is expected to be named as "class"
randomForestCrossValidation <- function(data) {
  require(randomForest)
  require(caret)
  errorrate_vec <- c()
  set.seed(11)
  folds <- createFolds(data$class, k = 10, list = TRUE, returnTrain=FALSE)
  for (i in 1:10) {
    test_idx <- unlist(folds[i])
    trainset <- data[-test_idx, ]
    testset <- data[test_idx, ]
    rf.fit <- randomForest(class ~ ., data = trainset,
                           ntree = 500, type = "classification")
    #print(rf.fit)
    predicted <- predict(rf.fit, newdata = testset, type = "response")
    errorrate <- sum(predicted != testset$class) / length(predicted)
    print(paste("Error rate is", errorrate))
    errorrate_vec <- c(errorrate_vec, errorrate)
  }
  errorrate_vec
}

## function that implements svm + 10 fold cross validation
## the response of the data is expected to be named as "class"
svmCrossValidation <- function(data) {
  require(e1071)
  require(caret)
  errorrate_vec <- c()
  set.seed(11)
  folds <- createFolds(data$class, k = 10, list = TRUE, returnTrain=FALSE)
  for (i in 1:10) {
    test_idx <- unlist(folds[i])
    trainset <- data[-test_idx, ]
    testset <- data[test_idx, ]
    svm.fit <- svm(class ~ ., data = trainset, type = "C")
    #print(rf.fit)
    predicted <- predict(svm.fit, newdata = testset)
    errorrate <- sum(predicted != testset$class) / length(predicted)
    print(paste("Error rate is", errorrate))
    errorrate_vec <- c(errorrate_vec, errorrate)
  }
  errorrate_vec
}

## function that implements logistic regression + 10 fold cross validation
## the response of the data is expected to be named as "class"
logisticRegressionCrossValidation <- function(data) {
  require(glm)
  require(caret)
  errorrate_vec <- c()
  set.seed(11)
  folds <- createFolds(data$class, k = 10, list = TRUE, returnTrain=FALSE)
  for (i in 1:10) {
    test_idx <- unlist(folds[i])
    trainset <- data[-test_idx, ]
    trainset$class <- ifelse(trainset$class=="A", TRUE, FALSE)
    testset <- data[test_idx, ]
    testset$class <- ifelse(testset$class=="A", TRUE, FALSE)
    #trainset$class <- (trainset$class=="A")
    lr.fit <- glm(as.factor(class) ~ ., data = trainset, family = binomial())
    #print(rf.fit)
    predictionsProb <- predict(lr.fit, testset, type="response")
    predicted <- (predictionsProb >= 0.5)
    errorrate <- sum(predicted != testset$class) / length(predicted)
    print(paste("Error rate is", errorrate))
    errorrate_vec <- c(errorrate_vec, errorrate)
  }
  errorrate_vec
}


## Step-0: Read in data
data <- read.csv(file="873examdataset2015.csv", header = FALSE, sep = ",")
class <- c(rep("A", 200), rep("B", 100))
#class <- c(rep("A", 100), rep("B", 200)) # false labeling injection
data[, 151] <- NULL
data <- cbind(class, data)

write.csv(data, file="data.csv", row.names=FALSE, quote=FALSE)

## variable clustering
require(Hmisc)
data_num <- data
data_num$class <- ifelse(data_num$class=="A", 1, 0)
# remove variables with no correlation with the reponse variable
corrs <- apply(data_num, 2, function(x) abs(cor(x, data_num$class)))
hist(corrs[-1], xlab="Correlation", main="Distribution of the variables' correlations \nwith the response (class)")
plot(corrs[-1], xlab = "Variable index", ylab="Correlation", 
     main="Each independent variable's correlations \nwith the response variable (class)")

corrs_top <- as.numeric(which(corrs > 0.45))
corrs_little <- as.numeric(which(corrs < 0.2))
v <- varclus(as.matrix(data_num[, -as.numeric(which(corrs < 0.3))]), 
             similarity="pearson", trans="abs",
             method="complete")
plot(v, cex=0.7, ylim=c(0,1))
#abline(h=0.4, col="red", lwd=3)

## correlation analysis
require(corrplot)
M <- cor(data_num[, corrs_top])
#corrplot(M, method="circle", cl.lim=c(0,1))

## Ignore variables with little correlation to response
data <- data[, -corrs_little]

## Step-1: PCA analysis: dimension reduction and data visulization
data_exp <- data[, -1]
pca <- prcomp(data_exp, scale=TRUE, center=TRUE) 
summary(pca)
screeplot(pca, type="l", npcs = 10,
          main="The variance of each principle component")
pch <- ifelse(data$class=="A", 1, 3)
plot(pca$x[,1], pca$x[,2], pch=pch, cex=1, xlab="PC1", ylab="PC2",
     xlim=c(-8, 8), ylim=c(-6,6), 
     main = "Scatter plot of the first two PCs")
legend("topright", title="Class", cex=0.8,
       legend=c("class A", "class B"), pch=c(1, 3))

data_pc10 <- cbind(class=data$class, as.data.frame(pca$x[, 1:10]))
data_pc2 <- cbind(class=data$class, as.data.frame(pca$x[, 1:2]))
data_pc98 <- cbind(class=data$class, as.data.frame(pca$x[, 1:98]))

## Step-2: Hierarhical clustering
# hierarchical clustering according to the measures other than crime rate
weighted_data <- data_pc98[, -c(1)]
weighted_data[, 2] <- weighted_data[, 2] * 1
d <- dist(as.matrix(weighted_data))   # find distance matrix 
hc <- hclust(d, method = "complete")             # apply hirarchical clustering 
plot(hc, main="Hierarchical Clusters of the Records", labels=FALSE) # plot the dendrogram
n_groups <- 2 # number of clusters
groups <- cutree(hc, k=n_groups) # cut tree into 3 clusters
# draw dendogram with red borders around the 3 clusters 
rect.hclust(hc, k=n_groups, border=1:4)

## Step-3: Label the groups on the PC scatter plot
plot(pca$x[,1], pca$x[,2], pch=pch, cex=1, xlab="PC1", ylab="PC2", col=groups,
     xlim=c(-8, 8), ylim=c(-6,6),
     main = "Scatter plot of the first two PCs")
legend("topright", title="Class A",
       legend=c("Cluster 1", "Cluster 2"),
       pch=c(1), col=c(1:2), cex=0.8)
legend("bottomright", title="Class B",
       legend=c("Cluster 1", "Cluster 2"),
       pch=c(3), col=c(1:2), cex=0.8)

# Locate the suspecious point
sus_points <- which(!(
  ((groups==1) & (data$class=="A")) |
    ((groups==2) & (data$class=="B"))
   ))
sus_points
# The 135th, 147th and 229th points are suspeciously mislabeling points

## Step-4: Random Forest on the raw data
rf_raw_results <- randomForestCrossValidation(data)
rf_raw_err <- mean(rf_raw_results)
rf_raw_err
hist(rf_raw_results, xlab="Error rate", xlim=c(0,0.05),
     main = "Distribution of error rates \nin 10-fold cross validation")

## svm on the raw data
svm_raw_results <- svmCrossValidation(data)
svm_raw_err <- mean(svm_raw_results)
svm_raw_err
hist(svm_raw_results, xlab="Error rate", xlim=c(0,0.05),
     main = "Distribution of error rates \nin 10-fold cross validation")

## Logistic regression
lr_raw_results <- logisticRegressionCrossValidation(data)
lr_raw_err <- mean(lr_raw_results)
lr_raw_err
hist(lr_raw_results, xlab="Error rate", xlim=c(0,0.5),
     main = "Distribution of error rates \nin 10-fold cross validation")

## Step-5: Random forest on dimension-reduced data
rf_red_results <- randomForestCrossValidation(data_pc2)
rf_red_err <- mean(rf_red_results)
rf_red_err

# ## Step-6: Detect outliers using random forest
# get_one_outlier <- function(data, draw=FALSE) {
#   set.seed(1)
#   rf.fit <- randomForest(class ~ ., data = data,
#                          ntree = 1000, type = "classification",
#                          proximity=TRUE)
#   pch <- ifelse(data$class=="A", 1, 3)
#   outlying <- outlier(rf.fit)
#   plot(outlying, pch=pch, ylab="Outlying measure", xlab="Record number")
#   #outlying_sorted <- sort(outlying, decreasing=TRUE)
#   #outlier_idx <- which(outlying >= outlying_sorted[70])
#   outlier_idx <- which.max(outlying)
#   as.numeric(c(outlying[outlier_idx], outlier_idx))
# }
# 
# outlier <- get_one_outlier(data, TRUE)
# outlier_value <- outlier[1]
# print(outlier_value)
# outlier_idx <- outlier[2]
# print(outlier_idx)
# data_int <- data
# while (outlier_value > 4) {
#   data_int <- data_int[-outlier_idx, ]
#   outlier <- get_one_outlier(data_int, FALSE)
#   outlier_value <- outlier[1]
#   print(outlier_value)
#   outlier_idx <- outlier[2]
#   print(outlier_idx)
# }

## TODO: correct the mislabling and re-run random forest
#new_data <- data[-outliers, ]
#new_data <- data_int
new_data <- data[-sus_points,]
#new_data[outlier_2, "class"] <- "B"
set.seed(1)
new_rf.fit <- randomForest(class ~ ., data = new_data,
                       ntree = 1000, type = "classification",
                       proximity=TRUE)
pch <- ifelse(new_data$class=="A", 1, 3)
outlying <- outlier(new_rf.fit)
plot(outlying, pch=pch, ylab="Outlying measure", xlab="Record number")

rf_new_results <- randomForestCrossValidation(new_data)
rf_new_err <- mean(rf_new_results)
rf_new_err
hist(rf_new_results, xlab="Error rate", xlim=c(0,0.05),
     main = "Distribution of error rates \nin 10-fold cross validation")


svm_new_results <- svmCrossValidation(new_data)
svm_new_err <- mean(svm_new_results)
svm_new_err

lr_new_results <- logisticRegressionCrossValidation(new_data)
lr_new_err <- mean(lr_new_results)
lr_new_err
