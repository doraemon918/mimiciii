df <- read.csv("~/Github/mimiciii/dfd.csv")
colnames(df)
df <- subset(df, select = -c(admittime, dischtime, ethnicity, insurance))
summary(df)
colnames(df)


# Drop least effective features 2/3 out of max, median, min
df <- subset(df, select = -c(subject_id, hadm_id, rdw_min, rdw_max, hemoglobin_min, hemoglobin_max, creatinine_median, creatinine_min, 
                             hematocrit_median, hematocrit_min, tempc_median, tempc_max, resprate_median, resprate_min, 
                             wbc_median, wbc_max, inr_median, inr_max, ptt_median, ptt_max, lactate_median, lactate_max, 
                             sysbp_median, sysbp_min, spo2_median, spo2_max, bilirubin_median, bilirubin_max, platelet_median, 
                             platelet_max, heartrate_max, heartrate_median))

# Rank

# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(followed_by_readmit~., data=df, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
print(importance)
# plot importance

# RFE

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(followed_by_readmit~., data=df, rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

png(filename="rplot3.png")
plot(importance)
dev.off()

### Naïve Bayes ###
trainIndex <- createDataPartition(df$followed_by_readmit, p = .8, 
                                  list = FALSE, 
                                  times = 1)

dfTrain <- df[ trainIndex,]
dfTest  <- df[-trainIndex,]

fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 

fit <- train(followed_by_readmit ~ ., 
              data=dfTrain, preprocessing = 'scale',
              method = 'rf',
              trControl = fitControl)

predicted <- predict(fit, dfTest)

confusionMatrix(predicted, dfTest$followed_by_readmit)

# AUROC
roc(as.integer(dfTest$followed_by_readmit), as.integer(predicted))
