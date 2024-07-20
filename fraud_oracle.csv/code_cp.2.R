#----------------------------Vehicle Insurance Claim Fraud Detection-----------------------

setwd("D:/DS CP/CP/fraud_oracle.csv")
f<-read.csv("fraud_oracle2.csv")
#categories for vehicle price------------
f$VehiclePrice<-as.factor(f$VehiclePrice)
#categories for days policy accidents ------------
f$Days_Policy_Accident<-as.factor(f$Days_Policy_Accident)
#categories for past  number of claim------------
f$PastNumberOfClaims<-as.factor(f$PastNumberOfClaims)
#categories for age of vehicle------------
f$AgeOfVehicle<-as.factor(f$AgeOfVehicle)
#categories for age of policy holder------------
f$AgeOfPolicyHolder<-as.factor(f$AgeOfPolicyHolder)
#categories for no of suppliment--------------
f$NumberOfSuppliments<-as.factor(f$NumberOfSuppliments)
#categories for addresschange claim----------
f$AddressChange_Claim<-as.factor(f$AddressChange_Claim)
#categories for mo of cars------------
f$NumberOfCars<-as.factor(f$NumberOfCars)
#categories for Policytype------------
f$PolicyType<-as.factor(f$PolicyType)
#categories for week of month claim------------
f$WeekOfMonthClaimed<-as.factor(f$WeekOfMonthClaimed)
#categories for Sex------------
f$Sex<-as.factor(f$Sex)
#categories for fraudfound----------------
f$FraudFound_P<-as.factor(f$FraudFound_P)
#categories for Martial Status------------
f$MaritalStatus<-as.factor(f$MaritalStatus)
#categories -------------------------------
f$Make<-as.factor(f$Make)
f$AccidentArea<-as.factor(f$AccidentArea)
f$MonthClaimed <- as.factor(f$MonthClaimed)
f$VehicleCategory <-as.factor(f$VehicleCategory)
f$Fault <-as.factor(f$Fault)
f$PoliceReportFiled <-as.factor(f$PoliceReportFiled)
f$WitnessPresent<-as.factor(f$WitnessPresent)
f$AgentType<-as.factor(f$AgentType)
f$BasePolicy<-as.factor(f$BasePolicy)
f$Days_Policy_Claim<-as.factor(f$Days_Policy_Claim)


# before sampling dataset----------------
cat("\n Before OverSampling:")
target_counts <- table(f$FraudFound_P)
print(target_counts)

#Dataset shuffle----------------------
set.seed(123)
shuffled_indices <- sample(nrow(f))
f <- f[shuffled_indices, ]

#-------balanced or unbalanced data---
#par(mar = c(5, 5, 2, 2))
#barplot(table(f$Make), main = "Class Distribution", xlab = "Class", ylab = "Frequency")
#barplot(table(f$VehicleCategory), main = "Class Distribution", xlab = "Class", ylab = "Frequency")

#otulier or not------------------
#library(ggplot2)
#p<-ggplot(f,aes(VehicleCategory,FraudFound_P))+geom_boxplot(outlier.colour = "red",outlier.shape = 4,outlier.size = 4)+theme(legend.position = "none")+labs(title="BOXPLOT")
#print(p)

#ROSE for oversampling------------------
cat("\n After OverSampling:")
#install.packages("ROSE")
library(ROSE)
f<- ROSE(as.formula("FraudFound_P ~ ."), data = f, seed = 123, p = 0.5)$data
table_result<- table(f$FraudFound_P)
print(table_result)

#write.csv(oversampled_data,"fraud_oracle_oversample.csv",row.names = FALSE)

#data partition----------------------------------------------
library(caret)
set.seed(123)
train_index <- createDataPartition(f$FraudFound_P, p = 0.75, list = FALSE)
training_data <- f[train_index, ]
testing_data <- f[-train_index, ]
training_data <-training_data[sample(nrow(training_data)), ]
testing_data <- testing_data[sample(nrow(testing_data)), ]


#decision tree Algorithm---------------------------------------------------------------------
#install.packages("randomForest")
library(caret)
ctrl <- trainControl(method = "cv",  number = 5, verboseIter = TRUE,search = "random")
grid <- data.frame(cp = 0.001)
model <- train(FraudFound_P ~ ., data = training_data, method = "rpart",  trControl = ctrl,  tuneGrid = grid, metric = "Accuracy")
print(model)
prediction_dt <- predict(model, testing_data[,colnames(testing_data)!="FraudFound_P"])
conf_matrix_dt<-confusionMatrix(prediction_dt, testing_data$FraudFound_P)
print(conf_matrix_dt)

#Random Forest algorithm-----------------------------------------------------------------------------
library(randomForest)
library(caret)
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE, search = "random")
tunegrid <- data.frame(mtry = 6)
model_rf <- train(FraudFound_P ~ .,  data = training_data, method = "rf", trControl = ctrl, tuneGrid = tunegrid, metric = "Accuracy",ntree = 700, nodesize = 5)
print(model_rf)
predictions_rf <- predict(model_rf, testing_data[, colnames(testing_data) != "FraudFound_P"])
conf_matrix_rf <- confusionMatrix(predictions_rf, testing_data$FraudFound_P)
print("Random Forest Confusion Matrix after Tuning:")
print(conf_matrix_rf)


# #NAive Bayes------------------------------------------------------------------------------------------
library(caret)
library(e1071)
train_control <- trainControl(method="cv", number=5)
tuning_grid <- expand.grid( .laplace = c(0, 0.5, 1),.usekernel = c(TRUE, FALSE),  .adjust = c(1))
model_nb <- train(FraudFound_P ~ ., data = training_data, method = "naive_bayes", trControl = train_control,tuneGrid = tuning_grid)
print(model)
predictions_nb <- predict(model, newdata = testing_data[,colnames(testing_data) != "FraudFound_P"])
conf_matrix_nb <- confusionMatrix(predictions_nb, testing_data$FraudFound_P)
print("Naive Bayes Confusion Matrix after Tuning:")
print(conf_matrix_nb)

#Logistic Regression-----------------------------------------------------------
library(caret)
train_control <- trainControl(method="cv", number=5)
hyperparameters <- expand.grid(alpha = 0:1,lambda =  0.001)
model_lr <- train(FraudFound_P ~ .,data = training_data,method = "glmnet",trControl = ctrl,tuneGrid = hyperparameters)
print(model_lr)
predictions_lr<- predict(model_lr, newdata = testing_data[,colnames(testing_data) != "FraudFound_P"])
conf_matrix_lr <- confusionMatrix(predictions_lr, testing_data$FraudFound_P)
print("Logistic Regression Confusion Matrix after Tuning:")
print(conf_matrix_lr)

#SVM --------------------------------------------------------------------
library(caret)
library(e1071)
train_control <- trainControl(method="cv", number=5)
hyperparameters <- expand.grid(C = c(0.1))
model_svm <- train(FraudFound_P ~ .,data = training_data,method = "svmLinear",trControl = ctrl,tuneGrid = hyperparameters,center=TRUE,scale=TRUE)
print(model_svm)
predictions_svm <- predict(model_svm, newdata = testing_data[,colnames(testing_data) != "FraudFound_P"])
conf_matrix_svm <- confusionMatrix(predictions_svm, testing_data$FraudFound_P)
print("SVM Confusion Matrix after Tuning:")
print(conf_matrix_svm)

#ROC Curve -------------------------------
library(pROC)
library(ggplot2)

# Calculate ROC curves for all models
roc_dt <- roc(testing_data$FraudFound_P, as.numeric(predict(model, testing_data)))
roc_rf <- roc(testing_data$FraudFound_P, as.numeric(predict(model_rf, testing_data)))
roc_nb <- roc(testing_data$FraudFound_P, as.numeric(predict(model_nb, testing_data)))
roc_lr <- roc(testing_data$FraudFound_P, as.numeric(predict(model_lr, testing_data)))
roc_svm <- roc(testing_data$FraudFound_P, as.numeric(predict(model_svm, testing_data)))

# Combine ROC curves into a dataframe
roc_data <- rbind(
  data.frame(Specificity = 1 - roc_dt$specificities, Sensitivity = roc_dt$sensitivities, Model = "Decision Tree"),
  data.frame(Specificity = 1 - roc_rf$specificities, Sensitivity = roc_rf$sensitivities, Model = "Random Forest"),
  data.frame(Specificity = 1 - roc_nb$specificities, Sensitivity = roc_nb$sensitivities, Model = "Naive Bayes"),
  data.frame(Specificity = 1 - roc_lr$specificities, Sensitivity = roc_lr$sensitivities, Model = "Logistic Regression"),
  data.frame(Specificity = 1 - roc_svm$specificities, Sensitivity = roc_svm$sensitivities, Model = "SVM")
)

# Calculate AUC values
auc_values <- c(auc(roc_dt), auc(roc_rf), auc(roc_nb), auc(roc_lr), auc(roc_svm))

# Define the positions for AUC labels
auc_positions <- data.frame(
  Model = c("Decision Tree", "Random Forest", "Naive Bayes", "Logistic Regression", "SVM"),
  Specificity = 0.8,
  Sensitivity = c(0.2, 0.3, 0.4, 0.5, 0.6)
)

# Plot ROC curves
ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Model)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "ROC Curves", x = "1 - Specificity", y = "Sensitivity") +
  scale_color_manual(values = c("blue", "red", "green", "purple", "orange")) +
  theme_minimal() +
  geom_text(data = auc_positions, aes(x = Specificity, y = Sensitivity, label = paste("AUC:", round(auc_values, 3))), color = "black", size = 3)


#Mutual Information---------------------------------------------------------
library(infotheo)
library(ggplot2)
 features <- f[, -12]
 target <- as.factor(f[[12]])
 discretize_numeric <- function(x, num_bins = 5) {
   cut(x, breaks = num_bins, labels = FALSE)
 }
 mi_values <- sapply(seq_along(features), function(i) {
  if (is.numeric(features[, i])) {
     discretized_feature <- discretize_numeric(features[, i])
     mutinformation(discretized_feature, target)
   } else {
     mutinformation(features[, i], target)
   }
 })
 mi_df <- data.frame(Feature = colnames(features), MI = mi_values)
sorted_mi_df <- mi_df[order(-mi_df$MI), ]

top_features <- sorted_mi_df$Feature[1:17]

p<-ggplot(data = sorted_mi_df, aes(x = reorder(Feature, -MI), y = MI)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Mutual Information Values for Features", x = "Features", y = "Mutual Information") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)

#Train Test Data for Fetaure selection----------------------------------------------
selected_data <- cbind(f[, c(top_features, "FraudFound_P")])
set.seed(123)
train_indices <- sample(nrow(selected_data), 0.75 * nrow(selected_data))
train_data <- selected_data[train_indices, ]
test_data <- selected_data[-train_indices, ]

#Decision Tree using feature selection---------------------------------------
library(rpart)
model_dt <- rpart(FraudFound_P ~ ., data = train_data, method = "class",cp=0.001)
predict_dt <- predict(model_dt, newdata = test_data[, colnames(test_data) != "FraudFound_P"], type = "class")
conf_matrixdt <- confusionMatrix(predict_dt, test_data$FraudFound_P)
print("Decision Tree Confusion Matrixusing mutual information:")
print(conf_matrixdt)


#RF using feature selelction----------------------------------------
library(randomForest)
modelrf <- randomForest(FraudFound_P ~ ., data = train_data,ntree=700,mtry=6)
predict_rf <- predict(modelrf, newdata = test_data[, colnames(test_data) != "FraudFound_P"])
confmatrix_fs <- confusionMatrix(predict_rf, test_data$FraudFound_P)
print("Random Forest Confusion Matrix using mutual information:")
print(confmatrix_fs)

#Roc AUC curve -------
library(ggplot2)
library(ROCR)

predictions_dt_numeric <- as.numeric(predict_dt)
predictions_rf_numeric <- as.numeric(predict_rf)

pred_dt <- prediction(predictions_dt_numeric, test_data$FraudFound_P)
pred_rf <- prediction(predictions_rf_numeric, test_data$FraudFound_P)

perf_dt <- performance(pred_dt, "tpr", "fpr")
perf_rf <- performance(pred_rf, "tpr", "fpr")

auc_dt <- performance(pred_dt, "auc")@y.values[[1]]
auc_rf <- performance(pred_rf, "auc")@y.values[[1]]

roc_data <- data.frame(
  FPR = c(perf_dt@x.values[[1]], perf_rf@x.values[[1]]),
  TPR = c(perf_dt@y.values[[1]], perf_rf@y.values[[1]]),
  Model = rep(c("Decision Tree", "Random Forest"), each = length(perf_dt@x.values[[1]]))
)

ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "ROC Curve - Decision Tree vs Random Forest",
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "Model"
  ) +
  annotate(
    "text",
    x = 0.5, y = 0.1,
    label = paste("Decision Tree AUC =", round(auc_dt, 3)),
    color = "blue"
  ) +
  annotate(
    "text",
    x = 0.5, y = 0.05,
    label = paste("Random Forest AUC =", round(auc_rf, 3)),
    color = "red"
  ) +
  theme_minimal()


##Mean decrease gini impurity--------------------------------------
library(randomForest)
library(ggplot2)
rf_model <- randomForest(f[, -12], f[,12])
feature_importance <- importance(rf_model)
importance_df <- data.frame(Feature = row.names(feature_importance),
                            MeanDecreaseGini = feature_importance[, "MeanDecreaseGini"],
                            row.names = NULL)
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]
g<-ggplot(importance_df, aes(x = MeanDecreaseGini, y = reorder(Feature, MeanDecreaseGini))) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  labs(title = "Mean Decrease in Gini Impurity for Features", x = "Mean Decrease in Gini Impurity", y = "Features") +
  theme(axis.text.y = element_text(angle = 0, hjust = 1))
print(g)

top_10_features <- head(importance_df$Feature, 19)
data_top_10 <- f[, c(top_10_features, "FraudFound_P")]
set.seed(123)
indices <- sample(nrow(data_top_10), 0.75 * nrow(data_top_10))
train <- data_top_10[indices, ]
test <- data_top_10[-indices, ]

library(rpart)
model_dt <- rpart(FraudFound_P ~ ., data = train, method = "class",cp=0.001)
predictdt <- predict(model_dt, newdata = test[, colnames(test) != "FraudFound_P"], type = "class")
conf_matrixdt <- confusionMatrix(predictdt, test$FraudFound_P)
print("Decision Tree Confusion Matrix using gini impurity:")
print(conf_matrixdt)

library(randomForest)
modelrf <- randomForest(FraudFound_P ~ ., data = train,ntree=700,mtry=6)
predictrf <- predict(modelrf, newdata = test[, colnames(test) != "FraudFound_P"])
confmatrix_fs <- confusionMatrix(predictrf, test$FraudFound_P)
print("Random Forest Confusion Matrix using gini impurity:")
print(confmatrix_fs)

library(e1071)
svm_model <- svm(FraudFound_P ~ ., data = train, kernel = "radial")
print(svm_model)
predictsvm<- predict(svm_model, newdata = test[, colnames(test) != "FraudFound_P"])
confmatrixsvm <- confusionMatrix(predictsvm, test$FraudFound_P)
print(" SVM  Confusion Matrix using gini impurity :")
print(confmatrixsvm)

# Calculate ROC and AUC for each model
roc_dt <- roc(test$FraudFound_P, as.numeric(predictdt))
roc_rf <- roc(test$FraudFound_P, as.numeric(predictrf))
roc_svm <- roc(test$FraudFound_P, as.numeric(predictsvm))

# Combine ROC curves into a dataframe
roc_data <- rbind(
  data.frame(Specificity = 1 - roc_dt$specificities, Sensitivity = roc_dt$sensitivities, Model = paste("Decision Tree (AUC:", round(auc(roc_dt), 3), ")")),
  data.frame(Specificity = 1 - roc_rf$specificities, Sensitivity = roc_rf$sensitivities, Model = paste("Random Forest (AUC:", round(auc(roc_rf), 3), ")")),
  data.frame(Specificity = 1 - roc_svm$specificities, Sensitivity = roc_svm$sensitivities, Model = paste("SVM (AUC:", round(auc(roc_svm), 3), ")"))
)

# Plot ROC curves
ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Model)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "ROC Curves", x = "1 - Specificity", y = "Sensitivity") +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme_minimal()


# Common features for shiny app
top_mi_features <- head(mi_df$Feature, 17)
top_gini_features <- head(importance_df$Feature, 19)
common_features <- intersect(top_mi_features, top_gini_features)
print(common_features)
data_common <- f[, c(common_features, "FraudFound_P")]
set.seed(123)
indices <- sample(nrow(data_common), 0.75 * nrow(data_common))
training <- data_common[indices, ]
testing <- data_common[-indices, ]

#Random forest using common features---------------------
library(randomForest)
model_rf_common <- randomForest(FraudFound_P ~ ., data = training, ntree = 700, mtry = 6)
saveRDS(model_rf_common, file = "random_forest_model.rds")
predictions_rf_common <- predict(model_rf_common, newdata = testing[, colnames(testing) != "FraudFound_P"])
conf_matrix_rf_common <- confusionMatrix(predictions_rf_common, testing$FraudFound_P)
print("Random Forest Confusion Matrix on Common Features:")
print(conf_matrix_rf_common)


#DT using common features---------------------------
library(randomForest)
model_dt_common <- rpart(FraudFound_P ~ ., data = training,method = "class",cp=0.001)
predictions_dt_common <- predict(model_dt_common, newdata = testing[, colnames(testing) != "FraudFound_P"],type="class")
conf_matrix_dt_common <- confusionMatrix(predictions_dt_common, testing$FraudFound_P)
print("DT Confusion Matrix on Common Features:")
print(conf_matrix_dt_common)


#SVM using common features-------------------------------
model_svm_common <- svm(FraudFound_P ~ ., data = training,kernal="radial")
predictions_svm_common <- predict(model_svm_common, newdata = testing[, colnames(testing) != "FraudFound_P"])
conf_matrix_svm_common <- confusionMatrix(predictions_svm_common, testing$FraudFound_P)
print("SVM Confusion Matrix on Common Features:")
print(conf_matrix_svm_common)

#ROC Curve for all agorithm contiaing common features-------
library(ROCR)
library(ggplot2)
predictions_rf_common <- predict(model_rf_common, newdata = testing[, colnames(testing) != "FraudFound_P"], type = "prob")[, 2]
predictions_dt_common <- predict(model_dt_common, newdata = testing[, colnames(testing) != "FraudFound_P"], type = "class")
predictions_svm_common <- predict(model_svm_common, newdata = testing[, colnames(testing) != "FraudFound_P"])
pred_rf <- prediction(predictions_rf_common, testing$FraudFound_P)
pred_dt <- prediction(ifelse(predictions_dt_common == "1", 1, 0), testing$FraudFound_P)
pred_svm <- prediction(ifelse(predictions_svm_common == "1", 1, 0), testing$FraudFound_P)

# Calculate TPR and FPR for each algorithm
perf_rf <- performance(pred_rf, "tpr", "fpr")
perf_dt <- performance(pred_dt, "tpr", "fpr")
perf_svm <- performance(pred_svm, "tpr", "fpr")

# Create data frames for ROC curve data
roc_data_rf <- data.frame(tpr = unlist(perf_rf@y.values), fpr = unlist(perf_rf@x.values), Model = "Random Forest")
roc_data_dt <- data.frame(tpr = unlist(perf_dt@y.values), fpr = unlist(perf_dt@x.values), Model = "Decision Tree")
roc_data_svm <- data.frame(tpr = unlist(perf_svm@y.values), fpr = unlist(perf_svm@x.values), Model = "SVM")
roc_data <- rbind(roc_data_rf, roc_data_dt, roc_data_svm)

# Calculate AUC for each algorithm
auc_value_rf <- as.numeric(performance(pred_rf, "auc")@y.values)
auc_value_dt <- as.numeric(performance(pred_dt, "auc")@y.values)
auc_value_svm <- as.numeric(performance(pred_svm, "auc")@y.values)

# Create data frame for AUC values
auc_data <- data.frame(Model = c("Random Forest", "Decision Tree", "SVM"),
                       AUC = c(auc_value_rf, auc_value_dt, auc_value_svm))

# Plot ROC curves using ggplot
ggplot(roc_data, aes(x = fpr, y = tpr, color = Model)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +  # Add diagonal line for reference
  geom_text(data = auc_data, aes(label = paste("AUC:", round(AUC, 3)), x = 0.8, y = c(0.2, 0.15, 0.1)), hjust = 1, vjust = 0, color = "black") +
  labs(title = "ROC Curves for Common Features", x = "False Positive Rate", y = "True Positive Rate") +
  scale_color_manual(values = c("Random Forest" = "blue", "Decision Tree" = "red", "SVM" = "green")) +
  theme_minimal()


