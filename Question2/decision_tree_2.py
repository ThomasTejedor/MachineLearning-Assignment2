#-------------------------------------------------------------------------
# AUTHOR: Thomas Christopher Tejedor
# FILENAME: decision_tree_2.py
# SPECIFICATION: Reads the given training data, test and output average performance of 10 tests. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['Question2/contact_lens_training_1.csv', 'Question2/contact_lens_training_2.csv', 'Question2/contact_lens_training_3.csv']
count = 1
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    X_mapping = [
        {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3},
        {'Myope': 1, 'Hypermetrope': 2},
        {'Yes': 1, 'No': 2},
        {'Normal': 1, 'Reduced': 2}
    ]

    Y_mapping = {'Yes': 1, 'No': 2}

    for row in dbTraining:
        age = X_mapping[0][row[0]]
        spectacle_prescription = X_mapping[1][row[1]]
        astigmatism = X_mapping[2][row[2]]
        tearProductionRate = X_mapping[3][row[3]]
        X.append([age,spectacle_prescription,astigmatism,tearProductionRate])
        
        Y.append(Y_mapping[row[4]])

    accuracy = 0
    #Loop your training and test tasks 10 times here
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        dbTest = []
        testCount = 0
        with open('Question2/contact_lens_test.csv', 'r') as testFile:
            reader = csv.reader(testFile)
            for j, row in enumerate(reader):
                if j > 0: #skipping the header
                    dbTest.append (row)
                    testCount += 1
        
        correctlyPredicted = 0
        
        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            testX_mapping = [
                {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3},
                {'Myope': 1, 'Hypermetrope': 2},
                {'Yes': 1, 'No': 2},
                {'Normal': 1, 'Reduced': 2}
            ]
            testY_mapping = {'Yes': 1, 'No': 2}
            
            class_predicted = clf.predict([[testX_mapping[0][data[0]], testX_mapping[1][data[1]], testX_mapping[2][data[2]], testX_mapping[3][data[3]]]])[0]
            
            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if(class_predicted == testY_mapping[data[4]]):
                correctlyPredicted += 1

        accuracy += (correctlyPredicted / testCount)
    #Find the average of this model during the 10 runs (training and test set)
    avg = accuracy / 10 
    
    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'final accuracy when training on contact_lens_training_{count}.csv:',avg)
    count +=1