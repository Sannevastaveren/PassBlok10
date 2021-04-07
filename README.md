# PassBlok10
Everything I need to do to pass blok 10
# Data Science:
notes from the all classes available still working on the last two of linear models and the Q&A
## Listen to Tilmans lectures in 2x speed!!! 1.5x is better for attention span, 2x just more efficient
# Image Analysis:
fun fact: my pc while running the code does about 1 step per second if done with gpu on colab it takes 240ms per step that is almost 75 % faster (if i understand how math works)
it did start way less accurate tho?? 50% accuracy at the start on gpu and 60% at the start on cpu


In ImageAnalysis/ouput you can find all models and plots except for cnn_1 and cnn_2 as they are above the 100mb limit of git pushes. Run code to get these takes about 1 hour to get both from colab. In the file Output.txt you can find 6 dictionaries. These dictionaries correspond to the values seen in the graphs for each model. 

> note that cnn_4_aug was done with 40 epochs and thus contains 20 more values in each array than the others

> note that the orange line is the test set and the blue line is the train set. If both lines are close to one another that means there is little over training as both get almost the same results

> note that the scale of each plot is different according to the values in the array. To properly compare all we need to make a constant X and Y axis (Y more so than X)
