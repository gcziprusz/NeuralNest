# curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/ship.csv
import csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# process the data
first = True
with open("ship.csv") as f:
    csvdata = csv.reader(f, delimiter=',')
    data = []
    for row in csvdata:
      if not first: data += [row]
      first = False

array = []
for col in range(len(data[0])):
  array += [{}]
  new = 0
  for i in range(len(data)):	
     line = data[i]		
     if line[col] not in array[col]:
      array[col][line[col]] = new
      new += 1  
alldat = []
alllabs = []
for line in data:
  alllabs += [int(line[1])]
  if line[5] == '': line[5] = '50'
  alldat += [ [int(line[2]), array[4][line[4]], float(line[5]), int(line[6]), int(line[7]), float(line[9]), line[11]=='S', line[11]=='C', line[11]=='Q' ]]
feats = ['Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked S', 'Embarked C', 'Embarked Q']

# randomly separate our training from test data 2/3 to 1/3
trainmask = [random.randint(0,2) for i in range(len(alldat))]

traindat = [alldat[i] for i in range(len(alldat)) if trainmask[i]<2]
trainlabs = [alllabs[i] for i in range(len(alldat)) if trainmask[i]<2]
testdat = [alldat[i] for i in range(len(alldat)) if trainmask[i]==2]
testlabs = [alllabs[i] for i in range(len(alldat)) if trainmask[i]==2]

# train a multi-layer perceptron with 60 hidden units to classify the data and print the accuracy.
nhidden = 60
clf = MLPClassifier(hidden_layer_sizes=[nhidden], max_iter = 50000)
clf = clf.fit(traindat, trainlabs)
pred = clf.predict(testdat)
[sum([pred[i] != testlabs[i] for i in range(len(testlabs))]) / len(testlabs)]

# calculate how much higher the predictions are for females vs. males
# feats = ['Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked S', 'Embarked C', 'Embarked Q']

imp = []
for v in alldat:
  real = v[1]
  v[1] = 0
  asmale = clf.predict_proba([v])[0][1]
  v[1] = 1
  asfemale = clf.predict_proba([v])[0][1]
  v[1] = real
  imp += [ asfemale-asmale ]

print("\nprediction for females over males")
print("-------------------------------------")
print(sum(imp)/len(imp))

# train a logistic regression and print the accuracy of the model on the train and test datasets
clf = LogisticRegression(max_iter = 500)

clf.fit(traindat, trainlabs)  

pred = clf.predict(traindat)
trainerr = sum([pred[i] != trainlabs[i] for i in range(len(trainlabs))]) / len(trainlabs)
pred = clf.predict(testdat)
testerr = sum([pred[i] != testlabs[i] for i in range(len(testlabs))]) / len(testlabs)

print("\naccuracy train and test")
print("-----------------------")
print(trainerr, testerr)

# plot the coefficients of the logistic regression model
print("\ncoefficients logistic regression")
print("--------------------------------")
for i in range(len(feats)):
  print(feats[i], clf.coef_[0][i])


