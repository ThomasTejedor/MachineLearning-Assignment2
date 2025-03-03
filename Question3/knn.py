#-------------------------------------------------------------------------
# AUTHOR: =Thomas Christopher Tejedor
# FILENAME: knn.py
# SPECIFICATION: Read email_classification.csv to filter spam using knn 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
samples = 0
incorrectPredictions = 0

#Reading the data in a csv file
with open('Question3/email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X = []
    Y = []
    testSample = []
    for row in db:
        if row != i:
            tempCol = []
            for col in range(20):
                tempCol.append(float(row[col]))
            X.append(tempCol)
            if(row[20] == 'spam'):
                Y.append(1.0)
            else:
                Y.append(0.0)
                
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    
    #Store the test sample of this iteration in the vector testSample
    for col in range(20):
        testSample.append(float(i[col]))
    
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict([testSample])[0]
    
    if(i[20] == 'spam'):
        testSample.append(1.0)
    else:
        testSample.append(0.0)

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if(class_predicted != testSample[20]):
        incorrectPredictions += 1
    samples += 1
    
#Print the error rate
errorRate = incorrectPredictions / samples
print('Error Rate:', errorRate)






