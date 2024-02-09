#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/vocab.txt
#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/spam-test.csv
#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/data/spam-train.csv

from sklearn import tree
import graphviz 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np

# read in the vocabulary file 
def readvocab(vocab_path="vocab.txt"):
   # keep track of the number of words
    lexiconsize = 0
   # initialize an empty dictionary
    word_dict = {}
   # create a feature for unknown words
    word_dict["@unk"] = lexiconsize
    lexiconsize += 1
   # read in the vocabular file
    with open(vocab_path, "r") as f:
        data = f.readlines()
   # Process the file a line at a time.
    for line in data:
        # The count is the first 3 characters
        count = int(line[0:4])
        # The word is the rest of the string
        token = line[5:-1]
       # Create a feature if it’s appeared at least twice
        if count > 1: 
            word_dict[token] = lexiconsize
            lexiconsize += 1
    # squirrel away the total size for later reference
    word_dict["@size"] = lexiconsize
    return(word_dict)
# Turn string str into a vector.
def tokenize(email_string, word_dict):
  # initially the vector is all zeros
  vec = [0 for i in range(word_dict["@size"])]
  # for each word
  for t in email_string.split(" "):
   # if the word has a feature, add one to the corresponding feature
    if t in word_dict: vec[word_dict[t]] += 1
   # otherwise, count it as an unk
    else: vec[word_dict["@unk"]] += 1
  return(vec)

# read in labeled examples and turn the strings into vectors
def getdat(filename, word_dict):
    with open(filename, "r") as f:
        data = f.readlines()
    dat = []
    labs = []
    for line in data:
        labs = labs + [int(line[0])]
        dat = dat + [tokenize(line[2:], word_dict)]
    return(dat, labs)

# train and test datasets, we'll build create the data and labels 
# we will use to train and use to test our naive Bayes model.
word_dict = readvocab()
traindat, trainlabs = getdat("spam-train.csv", word_dict)
testdat, testlabs = getdat("spam-test.csv", word_dict)

# fit a decision tree with 6 decision rules and print out the accuracy on the test data
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 6)	
clf = clf.fit(traindat, trainlabs)	

yhat = clf.predict(testdat)

sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)

# create a list of the words in our wordlist and use it to print the decision tree we have learned
wordlist = list(word_dict.keys())[:-1]
dot_data = tree.export_graphviz(clf, feature_names=wordlist,
                      filled=True, rounded=True) 
graph = graphviz.Source(dot_data)	
graph.view()

# How does the number of decision rules affect the accuracy of the model? 
# We'll retrain the model 29 times to see how the accuracy changes as we 
# increase the number of decision rules from 2 to 30.
for leaves in range(2, 31):
  clf = tree.DecisionTreeClassifier(max_leaf_nodes = leaves)	
  clf = clf.fit(traindat, trainlabs)	
  yhat = clf.predict(testdat)
  acc = sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)
  print(leaves,acc)

# Let's now fit a naive Bayes model and print the accuracy of the model
clf = MultinomialNB().fit(traindat, trainlabs)
clf = clf.fit(traindat, trainlabs)	
yhat = clf.predict(testdat)
acc = sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)
print(acc)

# We can also calculate the confusion matrix of the model, a table of the following counts:
# True Negatives False Positives
# False Negatives True Positives
print(confusion_matrix(testlabs, yhat))


# Let's visualize how Naive Bayes combines information from words in a sentence to make a judgement.
def plotsentence(sentence, clf):
  acc = 1.0
  labs = []
  facs = []
  factor = np.exp(clf.class_log_prior_[0]- clf.class_log_prior_[1])
  labs += ["PRIOR"]
  facs += [factor]
  acc *= factor
  for w in sentence:
    i = word_dict[w]
    factor = np.exp(clf.feature_log_prob_[0][i]- clf.feature_log_prob_[1][i])
    labs += [w]
    facs += [factor]
    acc *= factor
  labs += ["POST"]
  facs += [acc]
  return((labs,facs))

(labs,facs) = plotsentence(['yo', 'come', 'over', 'carlos', 'will', 'be', 'here', 'soon'], clf)
facs = [ fac if fac >= 1.0 else -1/fac for fac in facs ]
[(l,round(f,1)) for (l,f) in zip(labs,facs)]

(labs,facs) = plotsentence(['congratulations', 'thanks', 'to', 'a', 'good', 'friend', 'u', 'have', 'won'], clf)
facs = [ fac if fac >= 1.0 else -1/fac for fac in facs ]
[(l,round(f,1)) for (l,f) in zip(labs,facs)]