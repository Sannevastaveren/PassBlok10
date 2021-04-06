# Decision Trees
### Table example:
> this table is shortened

| AT%  |  -35box | -10box  |  -10ebox |  class |
|---|---|---|---|---|
|  81 |  yes | yes  |  no |  yes |
| 93  | yes  |  yes | no  | yes  |
| 60  |  yes |  no | no  | yes  |

> column 1 to 4 are features

> column 5 is the label/class

## **What are Decision trees?**

> Basically with a decision tree you're trying to find a pattern to check if certain data is part of a class or not this will be based on a training set. Once it's trained it will put test data into this decision tree and check this way if it is part of a class or not 
> - ofcourse this is not always the case as we probably don't know all features involved in the decision biologically so it is more an estimation therefor accuracy rate of a model is barely every 100% it is purely dependent on the knowledge of the person who made it or chance of putting in the right features
- has one root node (chosen feature to make a decision about)
- Branches connect to a node (the decision for instance yes or no or >30, <30)
- each branch goes to a leaf(class label the data has)
    * score looks like 0/3 (0 = number of errors, 3 = number of instances)
    > so there is 3 instances that have a -35box (since the branch was yes) and all of these have as class yes too therefor there are no errors. this can't be improved upon as there is no error
    * No does have error its score is 3/9
    > there is 9 instances that dont have a -35box yet 3 of them did have class as yes so this should go further by expanding the decision tree with a new feature from the **No** leaf same rules apply

- new node is another feature(-10ebxox) with branches attached to it with a choice yes or no, calculate the score, try to have 1 of the branches have a score with no errors.
    * choice yes or no -ebox10
    * 1/4 yes, 0/5 no (see that you only take the 9 instances that arent classified yet )
    * > always choose a tree that has minimal mistakes
    * > Now we have classified the no group, we can continue improving on the tree with the yes group 

- new node is another feature(AT%) this is a numeric value so the choice is above or below a value(=>45 or <= 45>) calculate the score of the leafs and try to get the least errors possible
    * choice => 45, <= 45
    * 0/3 =>45 , 0/1 <= 45
    * > All instances have now been classified now you hope that this will also apply to a not labeled dataset



## **How to compute Decision trees?**
> we are obviously not supposed to make these trees ourself by hand.. so how do we write an algorithm that does this

- Top down strategy
- Recursive divide and conquer
    * select attribute as root node
    * for each possibe attribute create a branch
        - split the instances into subsets one for each banch extending from the node
        - finally repeat recursively for each branch using only instances that reach the branch
    * stop if all instances are in a class

### **how to select the best attribute**
-   to get the smallest tree?
-   to get the "purest"(a branch gives a 100% correct score so like 0/3 is yes in the previous example, no errors) nodes?
#### Solution: **go for greates information gain!!!**

### **How to calculate information gain?**
> this wont be tested but is important to understand how it is related to the information gain (how to use entropy to calculate information)
- information (measured in bits)
    * entropy gives the information required in bits
    * formula for computing the entropy
        - entropy(p1,p2,..,pn) = -p1log(p1) - p2log(p2) ... - pnlog(pn)
        >(if there is two people and 2 rooms and both are in the same room) p is calculated by dividing how many are in the room and how many there are in total so for instance 2/2 = 1 so p1 is 1, and p2 is 0/2 = 0, 
        ```
        formula = -1log(1) - 0log(0) = 0
> Basically entropy measures if data is very seperate (he explains it with a room, if both are in the same room entropy is low, if they are seperate the entropy is high) therefor a low entropy would mean that there is few errors and thus is a good choice for an attribute

> Basics of Log :
> 2log(y) = x  --> 2^x = y
> - 2log(1) = 0
> - 2log(2) = 1
> - 2log(1/2) = -1
> - 2log(0) = 0

### Entropy:
- **Never** lower than 0
- if entropy is 0 it is very organized
- no maximum value (there can always be more chaos)
- max entropy is reached when all values are different categories
- minimum entropy is reached when all values are of the same category
- going from high entropy to low entropy costs energy (you cleaning your room makes tired but makes organized)

### Now how does this look in a decision tree?
> this takes all rows of the table and the class 
- First for the root info
    * formula : **-p1 x log(p1) - p2 x log(p2)**
    * p1 = yes = 6/12 = 0.5
    * p2 = no = 0.5 
        - Root : info([6(yes),6(no)]) = -0.5*log(0.5)-0.5*log(0.5) = 1
- Now for -35box
    * **total** info( [ 3,0 (yes) ],[ 3,6 (no) ] ) = 3/12 * 0 + 9/12 * 0.92 = 0.69
    > 9/12 instances it is the case that the info is 0.92, for 3/12 cases it the info is 0 the sum of that is the total info (remember low entropy is good and here we see that all values correct indeed gives entropy of 0)
    * **Yes** info( [3,0] ) = 1*log(1) - 0*log(0) = 0
    > 3/3 = 1, 0/3 = 0
    * **No** info( [3,6] ) = -0.33*log(0.33) - 0.66*log(0.66) = 0.92
    > 3/9 = 0.33, 6/9 = 0.66
    * **Gain**(-35box) = info(root) - info(-35box) = 1 - 0.69 = 0.31
- These steps you should follow for all features and then we choose the one with the highest gain as best option
- if you do this you get:
    * gain(-35box) = 1- 0.69 = 0.31
    * gain(-10ebox) = 1 - 0.9 = 0.1
    * gain(-10box) = 1- 0.97 - 0.03
    * > so obviously -35box is the best choice here
## **How to Refine Decision trees?**
- optimizing with data from train set may result in overfitting and wont work on other datasets 
    * Post-pruning methods can be uced to reduce this effect
### **When does overfitting occur?**
- Tree might be really complex and does not represent the problem because of irrelavent attributes
- values could be innaccurate
- if no generalization the algorithm could overfit the data

### **So why does post-pruning fix this?**
- pruning = deleting the end-nodes of the tree to prevent overfitting (could be done with an algorithm that calculates the best choice)
- the most relevant attribute for the problem is always at the top! so the bottom isnt as important
>  - this is because we calculate the choice that gives the most information thus what catagorizes most data into a class. Down the bottom its more extra little branches for the little exess data of side cases that arent as frequent. 

## **Summary**
- Trees classify data based on decision rules
- good for showing the relevance of features (if at the top important)
- can be visualized rather than being a black box
- starting with a tree is smart since it gives an idea of what features are important
- work well with categorical data (yes, no, male, female so on)