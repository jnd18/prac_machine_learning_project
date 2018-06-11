---
title: "Practical Machine Learning Final Project"
author: "Jonathan Dorsey"
date: "June 11, 2018"
output: 
    html_document:
        keep_md: yes
---


## Outline

In this report we detail the model building process we used to achieve over 99.9% accuracy in classifying the manner in which an exericse was performed using information from accelerometers. The model used was a random forest.


## Background
From the course project instructions:

> Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

In other words, we are given a data set of accelerometer measurements, and we must predict in which of five ways (A, B, C, D, or E) an exercise was performed.

## Preprocessing

First we download and load both the training set and the test set. The test set is set aside to the very end. Until then, when we mention "the data" we mean the training set.


```r
suppressPackageStartupMessages(library(tidyverse)) 
suppressPackageStartupMessages(library(caret)) 
suppressPackageStartupMessages(library(kableExtra)) 

train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


if (!file.exists("train.csv")) {
    download.file(train_url, "train.csv")
}
if (!file.exists("test.csv")) {
    download.file(test_url, "test.csv")
}

training <- read_csv("train.csv", col_types = cols())
testing <- read_csv("test.csv", col_types = cols())
```

Taking a quick look at the data, some columns appear to be almost entirely missing values. The table below shows the number of missing values (left column) and the number of columns that have that many missing values (right column).


```r
map(training, ~sum(is.na(.))) %>%
    as.numeric %>%
    table %>%
    as_data_frame %>%
    kable %>%
    kable_styling(full_width = F)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> . </th>
   <th style="text-align:right;"> n </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 0 </td>
   <td style="text-align:right;"> 57 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:right;"> 3 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19216 </td>
   <td style="text-align:right;"> 91 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19217 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19218 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19220 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19221 </td>
   <td style="text-align:right;"> 4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19293 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 19294 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>

So we'll remove those columns which have 19,000 or more missing values. After that, there are just three columns each with one missing value, so we can just omit any row with a missing value, and we'll lose at most only three rows. Also, we'll remove the first column because it just gives the row number. Since the classes come in order, i.e. all A's first, then all the B's, etc., the row number completely messes up the prediction algorithm later. Specifically, since the test set's rows are numbered 1 to 20, the algorithm will predict that the class should always be A, since low row numbers act as a perfect predictor of class A in the training data. Thus, we're left with just 59 predictors of the original 160.


```r
training <- training %>% select_if(~sum(is.na(.)) < 19000) %>% na.omit
```

Finally, we'll split a validation set off from the training data, so that we can get an assessment of model accuracy later. We'll use 80% of the data for training and 20% for validation.


```r
indices <- createDataPartition(training$classe, p = 0.8, list = FALSE)
val <- training[-indices, ]
training <- training[indices, ]
```

## Model Fitting and Results

Now we will train a random forest using the `ranger` package through `caret`. In the interest of speed, we use the "out-of-bag" or OOB resampling option to tune the model hyperparameters. This means that the performance of the random forest is assessed by having it classify each training point, but using only those trees whose bootstrapped sample did not contain that point. This means the random forest only has to be fit once with each hyperparameter combination as opposed to cross-validation which would require refitting the entire model 10 times for each configuration. The OOB estimate is generally comparable to cross-validation, and this built-in cross-validation substitute is one of the nice features of random forests.


```r
model <- train(classe ~ .,
               data = as.data.frame(training),
               method = "ranger",
               trControl = trainControl(method = "oob"),
               na.action = na.omit)
model
```

```
## Random Forest 
## 
## 15699 samples
##    59 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  splitrule   Accuracy   Kappa    
##    2    gini        0.9959870  0.9949241
##    2    extratrees  0.9756672  0.9691941
##   41    gini        0.9999363  0.9999194
##   41    extratrees  1.0000000  1.0000000
##   81    gini        0.9998726  0.9998389
##   81    extratrees  1.0000000  1.0000000
## 
## Tuning parameter 'min.node.size' was held constant at a value of 1
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were mtry = 41, splitrule =
##  extratrees and min.node.size = 1.
```

The print-out above shows the final settings. The hyperparameters were `mtry = 41`, `splitrule = extratrees`, and `min.node.size = 1`. The most important hyperparameter is probably `mtry` which is the number of predictors to choose from at each split. See the documentation of the `ranger` package for a full explanation.

We can also see that the accuracy is extremely high, actually 100%. This is the out of bag error, not the training error. Because the model seemed to be so accurate, we felt no need to try another model. So we went straight to testing the model on the validation set. 


```r
confusionMatrix(predict(model, val), as.factor(val$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  684    0    0
##          D    0    0    0  643    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

The confusion matrix above shows the results on the validation data. The accuracy was actually 100%, with a 95% confidence interval of (0.9991, 1). Thus, we are 95% confident that the model's out of sample error is at least 99.9%. This is obviously a huge improvement over the no information rate (i.e. expected accuracy from random guessing) of 28%. Of course, this should be taken with a small grain of salt because, for example, the IID assumption used to form this estimate probably doesn't hold exactly. Regardless, the model did correctly predict all 20 test cases, according to the prediction quiz. We consider this a highly successful model.

## Appendix

Below is all the session info needed to make the report fully reproducible. Also, the seed was set to 1234, and warnings were suppressed. The full `.Rmd` file is available in the repository.


```r
sessionInfo()
```

```
## R version 3.5.0 (2018-04-23)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 17134)
## 
## Matrix products: default
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] kableExtra_0.9.0 caret_6.0-80     lattice_0.20-35  forcats_0.3.0   
##  [5] stringr_1.3.0    dplyr_0.7.4      purrr_0.2.4      readr_1.1.1     
##  [9] tidyr_0.8.0      tibble_1.4.2     ggplot2_2.2.1    tidyverse_1.2.1 
## 
## loaded via a namespace (and not attached):
##  [1] httr_1.3.1         magic_1.5-8        ddalpha_1.3.3     
##  [4] viridisLite_0.3.0  sfsmisc_1.1-2      jsonlite_1.5      
##  [7] splines_3.5.0      foreach_1.4.4      prodlim_2018.04.18
## [10] modelr_0.1.1       assertthat_0.2.0   highr_0.6         
## [13] stats4_3.5.0       DRR_0.0.3          cellranger_1.1.0  
## [16] yaml_2.1.18        robustbase_0.93-0  ipred_0.9-6       
## [19] pillar_1.2.1       backports_1.1.2    glue_1.2.0        
## [22] digest_0.6.15      rvest_0.3.2        colorspace_1.3-2  
## [25] recipes_0.1.2      htmltools_0.3.6    Matrix_1.2-14     
## [28] plyr_1.8.4         psych_1.8.3.3      timeDate_3043.102 
## [31] pkgconfig_2.0.1    CVST_0.2-2         broom_0.4.4       
## [34] haven_1.1.1        scales_0.5.0       ranger_0.10.1     
## [37] gower_0.1.2        lava_1.6.1         withr_2.1.2       
## [40] nnet_7.3-12        lazyeval_0.2.1     cli_1.0.0         
## [43] mnormt_1.5-5       survival_2.41-3    magrittr_1.5      
## [46] crayon_1.3.4       readxl_1.1.0       evaluate_0.10.1   
## [49] nlme_3.1-137       MASS_7.3-49        xml2_1.2.0        
## [52] dimRed_0.1.0       foreign_0.8-70     class_7.3-14      
## [55] tools_3.5.0        hms_0.4.2          kernlab_0.9-26    
## [58] munsell_0.4.3      bindrcpp_0.2.2     e1071_1.6-8       
## [61] compiler_3.5.0     RcppRoll_0.3.0     rlang_0.2.0       
## [64] grid_3.5.0         iterators_1.0.9    rstudioapi_0.7    
## [67] rmarkdown_1.9      geometry_0.3-6     gtable_0.2.0      
## [70] ModelMetrics_1.1.0 codetools_0.2-15   abind_1.4-5       
## [73] reshape2_1.4.3     R6_2.2.2           lubridate_1.7.4   
## [76] knitr_1.20         utf8_1.1.3         bindr_0.1.1       
## [79] rprojroot_1.3-2    stringi_1.1.7      parallel_3.5.0    
## [82] Rcpp_0.12.16       rpart_4.1-13       tidyselect_0.2.4  
## [85] DEoptimR_1.0-8
```

