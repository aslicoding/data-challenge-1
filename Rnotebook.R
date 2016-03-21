library(RCurl)
library(corrplot)
library(ggplot2)
library(caret)
library(pROC)
library(plyr)

#Downloading the data

URL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data <- getURL(URL)
data.csv <- read.csv(textConnection(data),header=FALSE)
#check the dimensions
dim(data.csv)

#name the columns
colnames(data.csv)<-c("id","clump.thickness","uniformity.of.cell.size","uniformity.of.cell.shape",
                      "marginal.adhesion","single.epithelial.cell.size",
                      "bare.nuclei", "bland.chromatin","normal.nucleoli","mitosis","class")


#checking the column names
head(data.csv)

summary(data.csv)
#bare nuclei column has 56 observations classified as other, min and max values check out for the other ones
#take a closer look at the column bare nuclei
table(data.csv$bare.nuclei)
#there are 16 question marks as observations
#let's see if we'll need to convert these to a different variable or drop them as missing values
questiondat<-subset(data.csv,data.csv$bare.nucle=="?")
remainingdat<-subset(data.csv,data.csv$bare.nuclei !="?")
#let's look at the correlation structure of variables where bare nuclei are marked as ? and otherwise
#mitosis is selected out of the data set due to zero variance in the questiondat dataset
questiondat1<-subset(questiondat,select=-c(id, bare.nuclei, mitosis))
#gettin Pearson correlation
m<-cor(questiondat1)
corrplot(m, method=c("square"))
remainingdat1<-subset(remainingdat,select=-c(id, bare.nuclei, mitosis))
n<-cor(remainingdat1)
corrplot(n, method=c("square"))

#overall correlation structures are different with the caveat of sample size being small for the samples having ? at 'bare nuclei' field
#so we'll keep the question marks and convert them into a new variable
#we also see that features are  highly correlated with eachother. Multicollinearity is an issue for this data set
data.csv$bare.nuclei<-as.character(data.csv$bare.nuclei)
data.csv$bare.nuclei[data.csv$bare.nuclei=="?"]<- "11"
table(data.csv$bare.nuclei)
#all the question marks are replaced with 11
#let's also see if there are any repeated patients
length(data.csv$id)-length(unique(data.csv$id))
#54 patients are repeated. Let's see if these are duplicate measurements or multiple measurements for one patient
#makes up a little less than 1/6 of the data. Visual inspection shows that they are multiple measurements from the same patient.
temp<-data.csv$id[duplicated(data.csv$id)]
temp2<-data.csv[as.character(data.csv$id) %in% as.character(temp),]
temp2<-temp2[order(temp2$id),]
head(temp2)

#let's explore the relationship between benign and metastatic tumors and the features
ggplot(data.csv,aes(x=uniformity.of.cell.size, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=clump.thickness, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=uniformity.of.cell.size, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=uniformity.of.cell.shape, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=marginal.adhesion, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=marginal.adhesion, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=single.epithelial.cell.size, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=as.numeric(bare.nuclei), y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=bland.chromatin, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
ggplot(data.csv,aes(x=normal.nucleoli, y=as.numeric(class)))+geom_point(position=position_jitter(w=0.05,h=0.05))+geom_smooth()
#There are clear relationships between the features and the variables. As the scores increase, so does likelihood of being a metastatic cancer
#The PLS-DA algorithm can deal with colinearity
#partition the data
df<-subset(data.csv,select=-c(id))
df$bare.nuclei<-as.numeric(df$bare.nuclei)
df$class<-as.factor(df$class)
set.seed(985)
in_train <- createDataPartition(df$class, p=.80, list=FALSE)
training <- df[in_train, ]
testing <- df[-in_train, ]  

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

#need to convert the variable names for the class levels
feature.names=names(training)

for (f in feature.names) {
  if (class(training[[f]])=="factor") {
    levels <- unique(c(training[[f]]))
    training[[f]] <- factor(training[[f]],
                            labels=make.names(levels))
  }
}
feature.names=names(testing)
for (f in feature.names) {
  if (class(testing[[f]])=="factor") {
    levels <- unique(c(testing[[f]]))
    testing[[f]] <- factor(testing[[f]],
                           labels=make.names(levels))
  }
}


plsFit <- train(class ~ .,
                data = training,
                method = "pls",
                tuneLength = 15,
                trControl = ctrl,
                metric = "ROC",
                preProc = c("center", "scale"))

#Getting the ROC curves on the cross validated data set
predictorsNames <- names(testing)[names(testing) != 'class']
predictions <- predict(object=plsFit, training[,predictorsNames], type='prob')
head(predictions)
outcomeName <- 'class'
auc <- roc(ifelse(training[,"class"]=="X2",1,0), predictions[[1]])
print(auc$auc)
plot(auc)
plot(varImp(plsFit))
#Let's test the model on the holdout data set
predictpls<-predict(plsFit,testing)
confusionMatrix(predictpls,testing$class)

model_predictions <- predict(object=plsFit, testing[,predictorsNames], type='prob')
auc2 <- roc(ifelse(testing[,"class"]=="X2",1,0), model_predictions[[1]])

rocd<- data.frame(Sensitivity = auc2$sensitivities, Specificity = auc2$specificities)
print(auc2$auc)
  ggplot(rocd, aes(1-Specificity, Sensitivity)) + 
  geom_line(color = "cyan", cex = 1) + 
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = 2) 
