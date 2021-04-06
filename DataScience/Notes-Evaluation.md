# Evaluation

## properties train and test set

### **Error rate and stuff**
- Measuring a classifiers performace:
    * error rate = #errors / #instances

- Error rate on training set:
    * often very optimistic as it is trained to do it perfectly. this is called resubstitution error as it is calculated by resubstituting the training set into a classifier that you also trained your model iwth
- Therefor USE AN INDEPENDANT DATA SET to measure error rate
---------------------
### **All the sets:**
- Training set trains classifier
- Validation set optimizes parameters of classifier
- Test set used to calculate error rate of the final classifier(so after train and validation)
- test and train are both representative of the problem
- DONT USE TEST SET IN CREATING CLASSIFIER

### **Maximize use of data**
- bigger dataset is better classifier
- if you have a larger test set the error estimate is also more accurate as you cover more cases
> if I show you 10 people and ask you their name and you can name all 10 people your error estimate would be 0%. But does that mean if i show you 1  million people you'd still know everyone.. obviously not. yet if i show you 1 million people and you get 30% right somehow that would be a much more accurate number


## procedures to generate train and test set

### **Ratio and methods of dividing initial dataset**
- Hold out method(reserve instances for testing)
    - 2/3 of instances for training
    - 1/3 of instances for testing
- generate the sets by random sampling of instances(best method)
- generate stratified sets:
    * class distributions of the sets are the same as in the initial data

### **Generate train and test set**
- repeated holdout:
    * repeat holdout procedure (possible with stratification)
    * error rates are averaged to yield an overall rate
    * 50:50 split up of data and train and test tow times by swapping train and test set
    * ### **however this is not ideal since its better to train with more examples**

- Cross-Validation (form of repeated holdout)
    * instances are groups into partitions
    * every instance is used for testing exactly once
    * error rates are everaged
    * 10-fold-cross-validation gives the best estimate of error'
    * usefull for small dataset to make more of less
    > Basically you divide your dataset into for instance 3 parts then you make a model with 2 parts and test with 1 part. But then repeat this process but take a different part for testing eacht time and then estimate the error. This takes long.. so only really usefull when you have a small dataset

- Leave One Out Cross Validation
    - n-fold validation where n is the number of instances:
        *   greates possible amout of data useed for training
        * no random sampling
    - leave one out can not be stratified
    - > always use one part for testing

## Measuring Performance

### **Confusion Matrix**
Shows how many of what it got right and how many it got wrong. The higher the number the darker the color

| predicted 🠊 | red  | black  |
|---|---|---|
| red  |  4 |  1 |
| black  | 2  |  3 |
| actual 🠉  |   |   |

#### **What can I calculate from that?**
- true positive rate = TRP = TP/(TP/FN)
- false positive rate = FPR = FP/(FP/TN)
- success rate = (TP + TN)/ (TP+TN+FP+FN("all instances"))
- Sensitivity = TP/(TP+FN) = TPR
- Specificity = TN / (TN+FP) = 1-FNR
- Precision = TP/(TP+FP)

## ROC(Receiver Operating Characteristic) Curve

### **For what?**
- to visualize the performace of the classifier
- to compare different classifiers

### **What data goes into an ROC curve**
> a list sorted by the probability of the prediction

| instance  |  actual |  predicted |  probability |   |
|---|---|---|---|---|
| 1  |   yes|  yes | 0.99 |   |
| 20  |  yes |  yes |  0.98 |   |
| 3  | no  | yes  | 0.8  |   |

### **How does it look?**
- y axis true positive (0 to 100%)
- x axis false positive (0 to 100%)
- you go through the table if it was a true positive you go a step up if it was a false positive you go a step to the side. 
- the more linear your curve is the less predictive value your model is as its basically a 50/50 chance of yes and no
- supposed to look like a curve that first is very steep and then flattens the loser it gets to 100%
- The higher the curve the better the model can predict
- curve isn't always perfect some models fit better at the start some models fit better at the end so choosing a model depends on what you want to use the model for.
- Could combine classifiers to make an even better prediction

### **What can you calculate from it?**
- the area under the curve the more the better
