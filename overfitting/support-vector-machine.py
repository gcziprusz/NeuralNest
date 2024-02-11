# curl -LO https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
import random
from sklearn import svm
import matplotlib.pyplot as plt

# process the data:
with open("diabetes.csv", "r") as f:
    data = f.readlines()
feats = data[0]
feats = feats.replace('\n','')
feats = feats.split(",")
feats = feats[0:(len(feats)-1)]	
alldat = []				
alllabs = []			
for i in range(1,len(data)):	
     line = data[i]		
     line = line.replace('\n','')		
     csvline = line.split(",")		
     alllabs = alllabs + [int(csvline[len(csvline)-1])]
     csvline = [float(csvline[i]) for i in range(len(csvline)-1)]
     alldat = alldat + [csvline]
     
# We next will create a trainmask and use it to create the train, test data.
trainmask = [random.randint(0,2) for i in range(len(alldat))]

traindat = [alldat[i] for i in range(len(alldat)) if trainmask[i]]
trainlabs = [alllabs[i] for i in range(len(alldat)) if trainmask[i]]
testdat = [alldat[i] for i in range(len(alldat)) if not trainmask[i]]
testlabs = [alllabs[i] for i in range(len(alldat)) if not trainmask[i]]

# Fit 8 different classifier models using a support vector machine, 
# varying the degree of the polynomial from 0 to 7. 
# We then will calculate the training and test error of each model.
trainerr = []
testerr = []
degrees = [0,1,2,3,4,5,6,7]
for degree in degrees:
  clf = svm.SVC(gamma='scale',kernel='poly', degree=degree)
  clf.fit(traindat, trainlabs)  

  pred = clf.predict(traindat)
  trainerr += [sum([pred[i] != trainlabs[i] for i in range(len(trainlabs))]) / len(trainlabs)]
  pred = clf.predict(testdat)
  testerr += [sum([pred[i] != testlabs[i] for i in range(len(testlabs))]) / len(testlabs)]

# print the train and test error for the models by kernel degree.
plt.scatter(degrees, trainerr)
plt.plot(degrees, trainerr, label='training error')
plt.scatter(degrees, testerr)
plt.plot(degrees, testerr, label='test error')
plt.legend()
plt.title('Prediction error by polynomial kernel degree')
plt.show()