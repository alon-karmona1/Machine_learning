# Machine_learning
<br>

🏆**Machine Learning Project**🏆

The project is divided into 2 parts: Dataset Creation, and ML Models building and training (Decision Tree, Artificial Neural Networks, SVM, Unsupervised Learning - Clustering, Etc).

### 🏁 Part I-
In this section I practiced the first three steps in the process of creating a machine learning. Starting with Data Collection and Sensing, and continue with an emphasis on the third stage - Dataset Creation. These steps will be used in understanding and preparing the data for the second part in which I will use the data for training and examining models and comparing them. As part of Dataset Creation, I will show the probabilities (data distribution) of the target variable and other properties, the range of values each variable receives, present graphs showing my findings, and check if the data set is balanced.

<pre>
Project Database Explanation: 

The database contains information about computer games worldwide sales.
"X_test" - will be used for submission of the final prediction.
"X_train" - A file of explanatory variables with which i will try to predict the explained variable.
"Y_train" - Explained Variable File (EA_sales).
</pre>

### 🏁 Part II-
In this section I will use the data I prepared in Part A for the purpose of training and examining learning systems and comparing them. In this section I will show the connection of ML's theory to the practical writing of code using ready-made packages in PYTHON. For each model I will perform Hyperparameter Tuning to select the best configuration for it.
The models he applied in this section are-

✔**A. Decision Trees -**<br>
I will build a complete decision tree using the training set and check the percentages obtained on the training set and the validation set, and what can be deduced from these results. I will then train the decision tree with the best configuration obtained and check the accuracy percentages obtained on the training set and validation set, show a graph of the obtained tree, and explain what insights it provides on the problem and the different importance of the problem characteristics.

✔**B. Artificial Neural Networks -**<br>
Also in this model I will train and diagnose the neural network at the default values. After performing the Hyperparameter Tuning process to find the best configuration for this model, I will present the hyper-parameter values ​​tested as a function of the percentage of accuracy on the training and validation set, and explain for each hyper-parameter what the motivation was in choosing to tune it and .
Finally I will train the neural network using the selected configuration, perform a Binary classification task using a neural network and use the predict_proba function to get the output of the network (before it is converted to classes, i.e. the output of the last sigmoid layer of the network).

✔**C. SVM -**<br>
I will model SVM and perform Hyperparameter Tuning to find the best configuration, I will show the accuracy percentages of the model selected on the training and validation set, the separating straight equation, and I will understand the straight equation regarding the importance of each feature for classification.

✔**D. Unsupervised Learning - Clustering -**<br>
I will run the K-means algorithm and try different K values, I will check different from the previous models you ran and compare the cluster results for different K values that you tested. Find out what number of classes I will choose by the Silhouette Index, and the Davies - Bouldin Index.

✔**E. Comparison between the models - Evaluation**<br>
I will compare the performance of the three models (SVM, DT, MLP) and calculate the precision, recall, and F-1 scores for each model in order to examine what is the best model according to these indices, and what my conclusions are from it.

✔**F. Improving the chosen model - Improvements**<br>
Finally, I will offer an idea for improving the chosen model I have chosen. I will apply it and explain why I think it will improve your chosen model. I will check if the idea has really improved my model.

✔**G. Prediction**<br>
In the last part I will make a prediction using the model I chose on the attached "X_test.csv" data file. The predictions can be found in the file I attached as well.
 

 
 
