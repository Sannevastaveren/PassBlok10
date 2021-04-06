# Classification and Data

## Dataset (Raw Data)
Consists of: 
> Train(to make the model with labels) 

> Test/Validation set(benchmark to test/evaluate it's performance)

## Data wrangling/ Data munging
Transforming and mapping raw data into another format to make it more valuable and appropriote for the downstream process like an analysis (not everything is always important for every analysis)
> This takes up more time than the actual analysis usually :)

## Input Dataset
Class is already given as it has already been calculated beforehand so that the model trains on correct data and no wrong bias will get created

### consists of:
> X = observations & features

> Y = label (class) provided

### Table example:

|   | Feature 1  |  Feature 2 |  ... | Class  |
|---|---|---|---|---|
| Obs a  |   |   |   |  Yes |
| Obs b  |   |   |   |  No |
| Obs c  |   |   |   |  Yes |

> Observation(Obs) = object = instance

> Feature = attribute

> Class = label

## **Data Quality & Cleaning**
During pre processing we improve the quality by modifying the content by removing or correcting values

#### Example Quality Problems:
- Noise (innaccurate values)
- Missing values (NAN, Null)
- Duplicate data

### **Missing Values**
#### Why:
- not collected
- not relavent in all cases
#### How to handle:
- Remove instances
- Estimate 
- Ignore during analysis (not all algorithmes can handle this and will **crash**)

### **Duplicate Data**
**1** person **Many** mail addresses

>Creates bias in the model bc it occuress more often
#### Why:
- when merging from different sources
#### How to handle:
- Remove 


### **Data Transformation**
 
> choice of relevant attributes is important

> more attributes = more information

> not all are relevant and data can become too big (if many features)

### **Atrribute values**
categorical data is transformed into numeric(binary) data with for instance one hot encoding (turning word into 0s and 1s)
> Mapping to binary codes
- A = [1,0,0,0] C = [0,1,0,0] G = [0,0,1,0] T = [0,0,0,1]

    * Distance(A,C) = 1
        - wortel [ (1-0)2 + (0-1)2 + (0-0)2 + (0-0)2 ] 
        - = wortel[ 1 + 1] = wortel [ 2 ]
        - je vergelijkt alle getallen  dus 1 met 0 , 0 met 1, 0 met 0
    * Distance(A,G) = 1
> Distance with binary codes is always 1 !
> - formula: d(X,Y) = wortel[ (X-Y)2 ]
>   * d(X,Y,Z) = wortel[ (X-Y)2 + (Y-Z)2) ]

## Classification
### input:
- Data of observation vs features
- Preprocessing data:
    *   scaling(same as normalization)
    *   centering
    *   normalization(Could cause a dominant feature if not done, will overshadow others)
    *   handle missing values and outliers
- Class attribute is extra feature

## Summary:
>  - ### First Collect data
> - ### Data wrangling before building model
> - ### Use data visualisation to get insight into your data
> ### Ask Yourself:
> - which data / features are relavent?
> - what sources are used(is it a reliable data source will the data be correct?)
> - what is the quality (also depends on the source and methods of collecting the data)
> - is the data representative (is the data a good view of the problem, like do i only use prokaryoot then i will only use prokaryootic data as this will give different results in a model)

## Python Scripting:
### libraries:
- numpy
- matplotlib (.colors ListedColormap)
- sklearn (svm, metrics)

> class attribute as exrra vector in test set
```
train = np.array([[]],[[]])
class_train = [0,0,1,1,1]
test = np.array([[]],[[]])
class_test = [0,0,1,1,1]

svc = svm.SVC(kernel='linear')

# fit == train model
# train met train set en een vector met class voor idere instance
svs.fit(train, class_train)

# voorspelling van labels geeft vector

predicted = svc.predict(test)

# geeft score over hoe accuraat het model is (test met test set en checkt de class labels en berekend hiermee een score)
>
score = svc.score(test,class_test)

print "\n Score", score
print ""\n result overview\n
metrics.classification_report(class_test, predicted)
print "\n Confusion matrix: \n"
metrics.confusion_matrix(class_test, predicted)
```