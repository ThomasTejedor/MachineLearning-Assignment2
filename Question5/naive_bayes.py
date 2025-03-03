#-------------------------------------------------------------------------
# AUTHOR: Thomas Christopher Tejedor
# FILENAME: naive_bayes.py
# SPECIFICATION: Create a machine learning model using naive bayes using weather_training.csv
#   as a model, then use weather_test.py to test instances to find if tennis will be played 
#   and list instances with a confidence >= .75 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
dbTest = []
X = []
Y = []
#Reading the training data in a csv file
with open('Question5/weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if(i > 0): #skip the header 
            dbTraining.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

for row in dbTraining:
    X_mapping = [
        {'Sunny' : 1, 'Overcast' : 2, 'Rain' : 3},
        {'Hot' : 1, 'Mild' : 2, 'Cool' : 3},
        {'Normal' : 1, 'High' : 2},
        {'Weak' : 1, 'Strong' : 2}
    ]
    Y_mapping = {'Yes': 1, 'No': 2}
    
    
    outlook = X_mapping[0][row[1]]
    temperature = X_mapping[1][row[2]]
    humidity = X_mapping[2][row[3]]
    wind = X_mapping[3][row[4]]
    
    X.append([outlook,temperature,humidity,wind])
    Y.append(Y_mapping[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
with open('Question5/weather_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if(i > 0):
            dbTest.append(row)

for i, row in enumerate(dbTest):
    X_mapping = [
        {'Sunny' : 1, 'Overcast' : 2, 'Rain' : 3},
        {'Hot' : 1, 'Mild' : 2, 'Cool' : 3},
        {'Normal' : 1, 'High' : 2},
        {'Weak' : 1, 'Strong' : 2}
    ]
    Y_mapping = {'Yes': 1, 'No': 2}
    
    dbTest[i][1] = X_mapping[0][row[1]]
    dbTest[i][2] = X_mapping[1][row[2]]
    dbTest[i][3] = X_mapping[2][row[3]]
    dbTest[i][4] = X_mapping[3][row[4]]

#Printing the header of the solution
print(f'{'Day':<6} {'Outlook':<8} {'Temperature':<11} {'Humidity':<8} {'Wind':<6} {'PlayTennis':<10} {'Confidence'}')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in dbTest:
    #get test data
    currSample = []
    playTennis = ''
    for i in range (4):
        currSample.append(row[i+1])
    
    #predict probability for sample
    prob = clf.predict_proba([currSample])[0]
    
    highestProb = 0
    if prob[0] > prob[1]:
        playTennis = 'Yes'
        highestProb = prob[0]
    else:
        playTennis = 'No'
        highestProb = prob[1]
    
    if highestProb >= .75: 
        print(f'{row[0]:<6} {row[1]:<8} {row[2]:<11} {row[3]:<8} {row[4]:<6} {playTennis:<10} {highestProb:<10.2}')


