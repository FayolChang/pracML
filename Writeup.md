Predict activities using data collected from wearable devices
========================================================
This presentation shows how to build a model to predict activivities using machine learning algorithms. Generally,  there are four sections: loading the data, prepocessing the data, builing the model and  predicting using the model. 

## loading Data

```r
        trainURL = "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        testURL = "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!all(file.exists(c("training_raw.csv","testing_raw.csv"))))
{
        download.file(trainURL,destfile = "./training_raw.csv")
        download.file(testURL,destfile = "./testing_raw.csv")
}
        rm(trainURL,testURL)
training = read.csv("training_raw.csv")
testing20 = read.csv("testing_raw.csv")
```


## Preprocessing Data

In this section, Data is preprocessed for building model. The processing steps are follows:

1. the columns that contains too many NAs are elimated. 
2. the column 1 to 5 that are not useful to build the model, so they are not included.
3. columns that have near zero variance are removed.
4. columns that are highly corralated (corralation greater than 0.8) are discarded.
5. Data are transformed to zero means and variances eaqual to 1.


```r
require(caret)
# step 1.
# we first calcuate how many NAs in each column
# and exclude them in the data
num_NAs = apply(training,MARGIN=2,function(x)sum(is.na(x)))
cols_with_manyNA = names(num_NAs[num_NAs>0.3 * nrow(training)])
cols_ind = which(names(training) %in% cols_with_manyNA )

training_step1 = training[, -cols_ind]
testing20_step1 = testing20[, -cols_ind]

rm(num_NAs,cols_ind,cols_with_manyNA,training,testing20)
#rm(training,testing20)

# step 2.
# the first five columns are not related to the analysis
training_step2 = training_step1[, -c(1:5)]
testing20_step2 = testing20_step1[, -c(1:5)]

rm(training_step1,testing20_step1)


#step 3.
#remove features that have near zero variance.
zeroVar = nearZeroVar(training_step2)
training_step3 = training_step2[,-zeroVar]
testing20_step3 = testing20_step2[,-zeroVar]

rm(zeroVar,training_step2,testing20_step2)


# step 4.
# remove features that are highly related.
# NOTE:classe is the last column,so it will not affect
# the column index of highCorr
classe_ind = which(names(training_step3) %in% c("classe") )
descrCorr = cor(training_step3[,-classe_ind])
diag(descrCorr) = 0
highCorr = findCorrelation(descrCorr, 0.80)

training_step4 = training_step3[,-highCorr]
testing20_step4 = testing20_step3[,-highCorr]

rm(classe_ind,descrCorr,highCorr,training_step3,testing20_step3)




#step 5.
#centering and scaling
classe_ind = which(names(training_step4) %in% c("classe") )

training_step4[,-classe_ind] = scale(training_step4[,-classe_ind])
training_step5 = training_step4

testing20_step4[,-classe_ind] = scale(testing20_step4[,-classe_ind])
testing20_step5 = testing20_step4

rm(classe_ind,training_step4,testing20_step4)
```


## Building model

In this section,  steps of building model are shown. the data is splitted into training and testing data. and model was built using randomForest() function in randomForest package. Note there is no need to explicitly pass a cross-validation parameter to the function, because randomForest() do it internally.


```r
require(caret)
classe_ind = which(names(training_step5) %in% c("classe") )

train_index = createDataPartition(y=training_step5$classe,p=0.75,list=FALSE)
train_data = training_step5[train_index, ]
test_data = training_step5[-train_index, ]



require(randomForest)
set.seed(12345)
modelrf = randomForest(classe~.,data=train_data)

predrf = predict(modelrf,newdata = test_data[, -classe_ind])

confusionMatrix(predrf,test_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  943    2    0    0
##          C    0    2  853    5    0
##          D    0    0    0  799    3
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.994    0.998    0.994    0.997
## Specificity             0.999    0.999    0.998    0.999    1.000
## Pos Pred Value          0.997    0.998    0.992    0.996    1.000
## Neg Pred Value          1.000    0.998    1.000    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.183
## Detection Prevalence    0.285    0.193    0.175    0.164    0.183
## Balanced Accuracy       0.999    0.997    0.998    0.997    0.998
```

```r
rm(predrf,test_data,train_data,train_index,training_step5)
```



```r
predtest20 = predict(modelrf,newdata = testing20_step5[,-classe_ind])
predtest20
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  E  A  A  A  A  E  D  B  A  A  B  C  D  A  E  D  E  B  E  B 
## Levels: A B C D E
```




```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predtest20)
```
