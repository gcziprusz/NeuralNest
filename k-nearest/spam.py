#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/malware-test.csv
#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/malware-train.csv

from sklearn import neighbors
import matplotlib.pyplot as plt


# We then use this function to get the training and
#  test data used for our model.
def getdat(filename):
    with open(filename, "r") as f:
       data = f.readlines()
    dat = []
    labs = []
    for line in data:
        wordline = line.split(",")
        labs = labs + [wordline[0] == "pe-malicious"]
        dat = dat + [[float(wordline[i]) for i in range(1,len(wordline))]]
    return(dat,labs)
traindat, trainlabs = getdat("malware-train.csv")
testdat, testlabs = getdat("malware-test.csv")
# We use this testscore function to calculate the 
# accuracy of the model for four different values of k: 1, 5, 7, and 9
def testscore(dat,labs):
    yhats = clf.predict(dat)
    correct = sum([yhats[i] == labs[i] for i in range(len(dat))])
    return(correct)

acc = []
m = 4000
for k in [1,5,7,9]:
    clf = neighbors.KNeighborsClassifier(n_neighbors=k,metric="cosine")
    clf = clf.fit(traindat[:m], trainlabs[:m])
    acc = acc + [[k, testscore(traindat[:m],trainlabs[:m])/m, testscore(testdat,testlabs)/len(testlabs)]]

# Plot dependence on k
plt.plot([v[0] for v in acc], [v[1] for v in acc], c="blue")
plt.plot([v[0] for v in acc], [v[2] for v in acc], c="red")
plt.title('malware')
plt.xlabel('k')
plt.ylabel('acc')
plt.show()